#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# Workloads both implementations provide. The matched wgpu benchmark has no
# equivalent of the mixed families, so they are collected for Blade only and
# are used for the within-Blade barrier-scope comparison.
SHARED_WORKLOADS = (
    "compute-independent",
    "compute-chain",
    "graphics-independent",
    "graphics-chain",
)
BLADE_ONLY_WORKLOADS = (
    "mixed-independent",
    "mixed-chain",
)
BLADE_WORKLOADS = SHARED_WORKLOADS + BLADE_ONLY_WORKLOADS

# Placement crossed with scope. Neither axis is a default for the other, so
# every combination is collected and the comparison is symmetric.
VULKAN_POLICIES = (
    "automatic",
    "automatic-scoped",
    "hazard-only",
    "hazard-only-scoped",
    "explicit-all",
    "explicit-all-scoped",
)
# `manual_barriers` and `barrier_scope` are Vulkan concepts; Metal tracks
# hazards in the framework and has only one thing to measure.
METAL_POLICIES = ("automatic",)


def slugify(value: str, limit: int = 28) -> str:
    """Filesystem-safe fragment of a device name, for directory suffixes."""
    cleaned = []
    for character in value.lower():
        cleaned.append(character if character.isalnum() else "-")
    slug = "-".join(part for part in "".join(cleaned).split("-") if part)
    return slug[:limit].strip("-") or "device"


def host_label() -> str:
    return slugify(socket.gethostname().split(".")[0], limit=24)


def parse_pass_list(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(",") if part.strip())
    if not counts:
        raise argparse.ArgumentTypeError("--pass-list must name at least one count")
    if any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("--pass-list entries must be positive")
    return counts


def parse_arguments() -> argparse.Namespace:
    blade_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blade", type=Path, default=blade_root)
    parser.add_argument("--wgpu", type=Path, default=blade_root.parent / "wgpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--backend",
        choices=("vulkan", "metal"),
        default="metal" if sys.platform == "darwin" else "vulkan",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--blade-device-id", type=lambda value: int(value, 0))
    parser.add_argument("--wgpu-adapter-name")
    parser.add_argument("--passes", type=int, default=16)
    parser.add_argument(
        "--pass-list",
        type=parse_pass_list,
        help="comma-separated pass counts to sweep instead of a single --passes value",
    )
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    # Ten is not enough for a device that is still accelerating forty
    # iterations in; see the clock section of COLLECTING.md. A thousand costs
    # about a quarter of a second per block on a fast part and ten minutes
    # across a whole collection on the slowest one here.
    parser.add_argument("--warmups", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--allow-software", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-system-info", action="store_true")
    return parser.parse_args()


def run(
    command: list[str],
    cwd: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        rendered = shlex.join(command)
        raise RuntimeError(
            f"{rendered} failed in {cwd} with exit code {result.returncode}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result


def git_output(root: Path, *arguments: str) -> str:
    return run(["git", *arguments], root).stdout.strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_command_capture(
    output: Path, name: str, command: list[str], cwd: Path, timeout: int = 30
) -> None:
    if shutil.which(command[0]) is None:
        return
    try:
        result = run(command, cwd, timeout=timeout, check=False)
        content = (
            f"$ {shlex.join(command)}\n"
            f"exit_code={result.returncode}\n"
            f"{result.stdout}{result.stderr}"
        )
    except subprocess.TimeoutExpired as error:
        content = (
            f"$ {shlex.join(command)}\n"
            f"timeout={timeout}\n"
            f"{error.stdout or ''}{error.stderr or ''}"
        )
    (output / name).write_text(content, encoding="utf-8")


# Sysfs knobs that decide whether a device holds one clock for the length of a
# collection. A cell whose device changed clock state mid-block cannot resolve
# anything, and that is not visible in the timings afterwards -- only in the
# control floor, and only as an unexplained number. Recording the state makes a
# locked collection distinguishable from an unlocked one after the fact.
POWER_STATE_PATHS = (
    # amdgpu: `auto` is the default, `high` pins the top DPM level.
    "/sys/class/drm/card*/device/power_dpm_force_performance_level",
    # i915: min == max is a locked frequency.
    "/sys/class/drm/card*/gt_min_freq_mhz",
    "/sys/class/drm/card*/gt_max_freq_mhz",
    "/sys/class/drm/card*/gt_boost_freq_mhz",
    "/sys/class/drm/card*/gt_RP0_freq_mhz",
    # xe: the same idea under a different tree.
    "/sys/class/drm/card*/device/tile*/gt*/freq0/min_freq",
    "/sys/class/drm/card*/device/tile*/gt*/freq0/max_freq",
    "/sys/class/drm/card*/device/tile*/gt*/freq0/rp0_freq",
    "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
    "/sys/devices/system/cpu/intel_pstate/no_turbo",
)


def read_power_state() -> dict[str, str]:
    """Every readable clock-management knob, keyed by path."""
    if sys.platform != "linux":
        return {}
    state: dict[str, str] = {}
    for pattern in POWER_STATE_PATHS:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            try:
                state[str(path)] = path.read_text().strip()
            except OSError:
                continue
    return state


def unlocked_devices(state: dict[str, str]) -> list[str]:
    """Devices whose clocks are still under automatic management."""
    unlocked = []
    for path, value in state.items():
        if path.endswith("power_dpm_force_performance_level") and value == "auto":
            unlocked.append(f"{path} = auto")
    for path, value in state.items():
        if not path.endswith(("gt_min_freq_mhz", "freq0/min_freq")):
            continue
        maximum = state.get(
            path.replace("gt_min_freq_mhz", "gt_max_freq_mhz").replace(
                "freq0/min_freq", "freq0/max_freq"
            )
        )
        if maximum is not None and maximum != value:
            unlocked.append(f"{path} = {value}, max = {maximum}")
    return unlocked


def parse_metadata(output: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in output.splitlines():
        if not line.startswith("#"):
            continue
        fields = next(csv.reader([line[1:].strip()]))
        if len(fields) >= 2:
            metadata[fields[0].strip()] = ",".join(fields[1:]).strip()
    return metadata


def extract_benchmark_csv(output: str) -> str:
    """Remove validation-layer diagnostics that some backends print to stdout.

    Vulkan debug callbacks do not consistently use stderr. In particular,
    wgpu's callback can print a validation warning before the benchmark's CSV
    preamble. Keep the complete output in the validation log, but make the
    measurement file contain only metadata, its header, and data rows.
    """
    retained: list[str] = []
    data_columns: int | None = None
    for line in output.splitlines():
        if line.startswith("#"):
            retained.append(line)
            continue
        if line.startswith("sample,"):
            data_columns = len(next(csv.reader([line])))
            retained.append(line)
            continue
        if data_columns is None:
            continue
        fields = next(csv.reader([line]))
        if len(fields) != data_columns:
            continue
        try:
            int(fields[0])
        except ValueError:
            continue
        retained.append(line)
    return "\n".join(retained) + ("\n" if retained else "")


class Device:
    """One physical adapter, with the selector each implementation needs."""

    def __init__(self, name: str, blade_device_id: int | None, software: bool) -> None:
        self.name = name
        self.blade_device_id = blade_device_id
        self.software = software

    @property
    def slug(self) -> str:
        return slugify(self.name)

    def __repr__(self) -> str:
        return f"Device({self.name!r}, id={self.blade_device_id}, software={self.software})"


def list_blade_adapters(binary: Path, cwd: Path) -> list[Device]:
    """Parse `sync-bench --list-adapters`: id, name, software=bool, status."""
    result = run([str(binary), "--list-adapters"], cwd, check=False)
    if result.returncode != 0:
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        try:
            device_id = int(fields[0], 0)
        except ValueError:
            continue
        software = fields[2].strip().endswith("true")
        devices.append(Device(fields[1].strip(), device_id, software))
    return devices


def list_wgpu_adapter_names(binary: Path, cwd: Path, backend: str) -> set[str]:
    """Parse `wgpu-sync-bench --list-adapters`: ids, name, type, driver, info."""
    environment = os.environ.copy()
    environment["WGPU_BACKEND"] = backend
    result = run([str(binary), "--list-adapters"], cwd, env=environment, check=False)
    if result.returncode != 0:
        return set()
    names = set()
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            names.add(fields[1].strip())
    return names


def discover_devices(
    blade_binary: Path,
    wgpu_binary: Path,
    cwd: Path,
    arguments: argparse.Namespace,
) -> list[Device]:
    """Devices to collect on, most specific request first.

    An explicit `--blade-device-id` or `--wgpu-adapter-name` pins a single
    device and keeps the previous single-collection behaviour. Otherwise every
    adapter both implementations can see is collected in turn, because a
    machine with an integrated and a discrete GPU has two answers, not one.
    """
    if arguments.blade_device_id is not None or arguments.wgpu_adapter_name:
        return [
            Device(
                arguments.wgpu_adapter_name or "pinned",
                arguments.blade_device_id,
                software=False,
            )
        ]

    blade_devices = list_blade_adapters(blade_binary, cwd)
    if not blade_devices:
        # Backends without adapter enumeration (or a listing failure) still
        # collect once, on whatever each implementation picks by default.
        print(
            "warning: no adapters enumerated; collecting once on the default device",
            file=sys.stderr,
        )
        return [Device("default", None, software=False)]

    wgpu_names = list_wgpu_adapter_names(wgpu_binary, cwd, arguments.backend)
    selected = []
    for device in blade_devices:
        if device.software and not arguments.allow_software:
            print(f"skipping software device: {device.name}", file=sys.stderr)
            continue
        if wgpu_names and device.name not in wgpu_names:
            print(
                f"skipping {device.name}: not visible to wgpu on this backend",
                file=sys.stderr,
            )
            continue
        selected.append(device)
    if not selected:
        raise ValueError(
            "no device is visible to both implementations; pass --blade-device-id "
            "and --wgpu-adapter-name to pin one explicitly"
        )
    return selected


def ensure_positive(arguments: argparse.Namespace) -> None:
    values = {
        "repetitions": arguments.repetitions,
        "passes": arguments.passes,
        "elements": arguments.elements,
        "rounds": arguments.rounds,
        "width": arguments.width,
        "height": arguments.height,
        "samples": arguments.samples,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if arguments.warmups < 0:
        raise ValueError("--warmups must not be negative")


def collect_device(
    *,
    device: Device,
    output: Path,
    blade: Path,
    wgpu: Path,
    blade_binary: Path,
    wgpu_binary: Path,
    collection_id: str,
    repository_metadata: dict[str, dict[str, str]],
    seed: int,
    arguments: argparse.Namespace,
) -> None:
    """Run the full randomized matrix for one device into its own collection."""
    if output.exists():
        raise ValueError(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    equivalence = run(
        [
            sys.executable,
            str(blade / "paper/check-workload-equivalence.py"),
            "--blade",
            str(blade),
            "--wgpu",
            str(wgpu),
        ],
        blade,
    )
    (output / "workload-shaders.txt").write_text(equivalence.stdout, encoding="utf-8")

    manifest: dict[str, object] = {
        "schema": "blade-sync-study-v1",
        "collection_id": collection_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "backend": arguments.backend,
        "seed": seed,
        "requested_device": {
            "name": device.name,
            "blade_device_id": device.blade_device_id,
            "slug": device.slug,
        },
        "repositories": repository_metadata,
        "parameters": {
            "repetitions": arguments.repetitions,
            "passes": arguments.passes,
            "pass_list": list(arguments.pass_list) if arguments.pass_list else None,
            "elements": arguments.elements,
            "rounds": arguments.rounds,
            "width": arguments.width,
            "height": arguments.height,
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "validation": arguments.validation,
            "gpu_timing": not arguments.cpu_only,
            "allow_software": arguments.allow_software,
            "blade_device_id": device.blade_device_id,
            "wgpu_adapter_name": device.name if device.name != "default" else None,
        },
        "runs": [],
    }
    (output / "blade-status.txt").write_text(
        git_output(blade, "status", "--short", "--branch") + "\n", encoding="utf-8"
    )
    (output / "wgpu-status.txt").write_text(
        git_output(wgpu, "status", "--short", "--branch") + "\n", encoding="utf-8"
    )
    if not arguments.skip_system_info:
        write_command_capture(
            output, "rustc.txt", ["rustc", "--version", "--verbose"], blade
        )
        write_command_capture(
            output, "cargo.txt", ["cargo", "--version", "--verbose"], blade
        )
        if arguments.backend == "vulkan":
            write_command_capture(
                output, "vulkaninfo.txt", ["vulkaninfo", "--summary"], blade
            )
            write_command_capture(
                output, "vulkaninfo-full.txt", ["vulkaninfo"], blade, timeout=60
            )
            write_command_capture(output, "nvidia-smi.txt", ["nvidia-smi", "-q"], blade)
            write_command_capture(
                output, "rocm-smi.txt", ["rocm-smi", "--showallinfo"], blade
            )
        if sys.platform == "darwin":
            write_command_capture(
                output,
                "system-profiler.txt",
                ["system_profiler", "SPHardwareDataType", "SPDisplaysDataType"],
                blade,
                timeout=60,
            )
            write_command_capture(output, "power.txt", ["pmset", "-g", "custom"], blade)
        if sys.platform == "win32":
            write_command_capture(
                output, "systeminfo.txt", ["systeminfo"], blade, timeout=60
            )

    power_state = read_power_state()
    manifest["power_state"] = power_state
    manifest["clocks_locked"] = bool(power_state) and not unlocked_devices(power_state)
    if power_state:
        (output / "power-state.txt").write_text(
            "".join(f"{path} = {value}\n" for path, value in power_state.items()),
            encoding="utf-8",
        )
        unlocked = unlocked_devices(power_state)
        if unlocked:
            print(
                "warning: GPU clocks are under automatic management:\n  "
                + "\n  ".join(unlocked)
                + "\n  Short command buffers on an idle device then change clock "
                "state mid-block, which raises the control floor until the cell "
                "answers nothing. See the clock-locking section of "
                "COLLECTING.md.",
                file=sys.stderr,
            )

    blade_policies = (
        VULKAN_POLICIES if arguments.backend == "vulkan" else METAL_POLICIES
    )
    blade_workloads = (
        BLADE_WORKLOADS if arguments.backend == "vulkan" else SHARED_WORKLOADS
    )
    pass_counts = arguments.pass_list or (arguments.passes,)
    configurations = [
        ("blade", workload, policy, passes)
        for passes in pass_counts
        for workload in blade_workloads
        for policy in blade_policies
    ]
    configurations.extend(
        ("wgpu", workload, "tracked", passes)
        for passes in pass_counts
        for workload in SHARED_WORKLOADS
    )
    rng = random.Random(seed)
    common_arguments = [
        "--elements",
        str(arguments.elements),
        "--rounds",
        str(arguments.rounds),
        "--width",
        str(arguments.width),
        "--height",
        str(arguments.height),
        "--warmups",
        str(arguments.warmups),
        "--samples",
        str(arguments.samples),
    ]
    if arguments.validation:
        common_arguments.append("--validation")
    if arguments.cpu_only:
        common_arguments.append("--no-gpu-timing")
    if arguments.allow_software:
        common_arguments.append("--allow-software")

    order_path = output / "order.csv"
    validation_hashes: dict[tuple[int, str, int], set[str]] = {}
    device_names: dict[str, set[str]] = {}
    with order_path.open("w", encoding="utf-8", newline="") as order_file:
        order_writer = csv.writer(order_file)
        order_writer.writerow(
            (
                "repetition",
                "index",
                "implementation",
                "workload",
                "policy",
                "passes",
                "file",
            )
        )
        for repetition in range(1, arguments.repetitions + 1):
            order = list(configurations)
            rng.shuffle(order)
            for index, (implementation, workload, policy, passes) in enumerate(order):
                run_id = f"r{repetition:02d}__{implementation}__{workload}__{policy}"
                if len(pass_counts) > 1:
                    run_id = f"{run_id}__p{passes:04d}"
                csv_name = f"{run_id}.csv"
                command = [
                    str(blade_binary if implementation == "blade" else wgpu_binary),
                    "--workload",
                    workload,
                    "--policy",
                    policy,
                    "--passes",
                    str(passes),
                    *common_arguments,
                ]
                environment = os.environ.copy()
                relevant_environment: dict[str, str] = {}
                if arguments.validation and arguments.backend == "vulkan":
                    # Force the Khronos layer to load and enable its
                    # synchronization checks. If the layer is unavailable the
                    # benchmark fails instead of silently producing a
                    # "validation" collection that validated nothing.
                    environment["VK_INSTANCE_LAYERS"] = (
                        "VK_LAYER_KHRONOS_validation"
                    )
                    # Current validation layers warn that the older
                    # `VK_LAYER_ENABLES=...SYNCHRONIZATION_VALIDATION_EXT`
                    # setting is deprecated. Use the corresponding named
                    # setting so correctness logs stay warning-free.
                    environment.pop("VK_LAYER_ENABLES", None)
                    environment["VK_LAYER_VALIDATE_SYNC"] = "1"
                    relevant_environment["VK_INSTANCE_LAYERS"] = environment[
                        "VK_INSTANCE_LAYERS"
                    ]
                    relevant_environment["VK_LAYER_VALIDATE_SYNC"] = environment[
                        "VK_LAYER_VALIDATE_SYNC"
                    ]
                if implementation == "blade" and device.blade_device_id is not None:
                    command.extend(["--device-id", str(device.blade_device_id)])
                if implementation == "wgpu":
                    environment["WGPU_BACKEND"] = arguments.backend
                    relevant_environment["WGPU_BACKEND"] = arguments.backend
                    if device.name not in ("default", "pinned"):
                        environment["WGPU_ADAPTER_NAME"] = device.name
                        relevant_environment["WGPU_ADAPTER_NAME"] = device.name
                    elif arguments.wgpu_adapter_name:
                        environment["WGPU_ADAPTER_NAME"] = arguments.wgpu_adapter_name
                        relevant_environment["WGPU_ADAPTER_NAME"] = (
                            arguments.wgpu_adapter_name
                        )

                print(f"{output.name}/{run_id}", file=sys.stderr, flush=True)
                result = run(command, blade, env=environment, timeout=900, check=False)
                if result.stderr:
                    (output / f"{run_id}.stderr.txt").write_text(
                        result.stderr, encoding="utf-8"
                    )
                validation_output = result.stdout + "\n" + result.stderr
                validation_log: Path | None = None
                if arguments.validation:
                    validation_log = output / f"{run_id}.validation.txt"
                    validation_log.write_text(validation_output, encoding="utf-8")
                validation_errors = [
                    line
                    for line in validation_output.splitlines()
                    if "SYNC-HAZARD" in line or "Validation Error" in line
                ]
                if arguments.validation and validation_errors:
                    assert validation_log is not None
                    raise RuntimeError(
                        f"{run_id} reported synchronization/validation errors; "
                        f"full combined output is in {validation_log.name}\n"
                        + "\n".join(validation_errors[:20])
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{run_id} failed with exit code {result.returncode}\n"
                        f"{result.stdout}{result.stderr}"
                    )
                captured = (
                    f"# collection_id,{collection_id}\n"
                    f"# repetition,{repetition}\n"
                    f"{extract_benchmark_csv(result.stdout)}"
                )
                (output / csv_name).write_text(captured, encoding="utf-8")
                metadata = parse_metadata(captured)
                expected = {
                    "schema": "blade-sync-bench-v1",
                    "implementation": implementation,
                    "backend": arguments.backend,
                    "validation": str(arguments.validation).lower(),
                    "gpu_timing": str(not arguments.cpu_only).lower(),
                }
                for key, value in expected.items():
                    if metadata.get(key) != value:
                        raise ValueError(
                            f"{run_id}: {key} is {metadata.get(key)!r}, expected {value!r}"
                        )
                validation_hash = metadata.get("validation_hash")
                if validation_hash is None:
                    raise ValueError(f"{run_id}: missing validation hash")
                validation_hashes.setdefault(
                    (repetition, workload, passes), set()
                ).add(validation_hash)
                device_names.setdefault(implementation, set()).add(
                    metadata.get("device_name", "")
                )
                order_writer.writerow(
                    (
                        repetition,
                        index,
                        implementation,
                        workload,
                        policy,
                        passes,
                        csv_name,
                    )
                )
                run_record = {
                    "id": run_id,
                    "file": csv_name,
                    "command": command,
                    "environment": relevant_environment,
                    "metadata": metadata,
                }
                if validation_log is not None:
                    run_record["validation_log"] = validation_log.name
                manifest["runs"].append(run_record)

    for (repetition, workload, passes), hashes in validation_hashes.items():
        if len(hashes) != 1:
            raise ValueError(
                f"repetition {repetition} workload {workload} with {passes} passes "
                f"produced different validation hashes: {sorted(hashes)}"
            )

    # A matched comparison is meaningless when the two implementations pick
    # different physical devices, which is easy to do on a machine that has both
    # an integrated and a discrete GPU. Fail loudly rather than publish it.
    selected = {name for names in device_names.values() for name in names}
    if len(selected) != 1:
        raise ValueError(
            "implementations selected different devices: "
            + "; ".join(
                f"{implementation}={sorted(names)}"
                for implementation, names in sorted(device_names.items())
            )
            + "\nPass --blade-device-id and --wgpu-adapter-name to pin one device."
        )
    if arguments.validation:
        manifest["validation_scan"] = {
            "khronos_synchronization_validation": arguments.backend == "vulkan",
            "error_patterns": ["SYNC-HAZARD", "Validation Error"],
            "errors_found": 0,
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Raw results: {output}", file=sys.stderr)


def main() -> None:
    arguments = parse_arguments()
    ensure_positive(arguments)
    blade = arguments.blade.resolve()
    wgpu = arguments.wgpu.resolve()
    collection_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not (blade / "Cargo.toml").is_file():
        raise ValueError(f"not a Blade checkout: {blade}")
    if not (wgpu / "examples/standalone/sync_bench/Cargo.toml").is_file():
        raise ValueError(f"wgpu sync benchmark is missing from: {wgpu}")

    repositories = {"blade": blade, "wgpu": wgpu}
    repository_metadata: dict[str, dict[str, str]] = {}
    for name, root in repositories.items():
        # Accumulated collections do not change what the binaries do, so they
        # do not make the source revision ambiguous. Everything else does.
        status = git_output(
            root, "status", "--porcelain", "--", ":(exclude)paper/data"
        )
        if status and not arguments.allow_dirty:
            raise ValueError(
                f"{name} worktree is dirty; commit it or pass --allow-dirty for a pilot"
            )
        repository_metadata[name] = {
            "root": str(root),
            "revision": git_output(root, "rev-parse", "HEAD"),
            "branch": git_output(root, "branch", "--show-current"),
            "status": status,
            "cargo_lock_sha256": file_sha256(root / "Cargo.lock"),
        }

    if not arguments.skip_build:
        run(["cargo", "build", "--release", "--example", "sync-bench"], blade)
        run(["cargo", "build", "--release", "-p", "wgpu-sync-bench"], wgpu)

    blade_binary = blade / "target/release/examples/sync-bench"
    wgpu_binary = wgpu / "target/release/wgpu-sync-bench"
    if sys.platform == "win32":
        blade_binary = blade_binary.with_suffix(".exe")
        wgpu_binary = wgpu_binary.with_suffix(".exe")
    if not blade_binary.is_file() or not wgpu_binary.is_file():
        raise ValueError("release benchmark binaries are missing")

    seed = arguments.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little")

    devices = discover_devices(blade_binary, wgpu_binary, blade, arguments)
    for device in devices:
        print(f"device: {device}", file=sys.stderr)

    # `<collection-id>-<hostname>`, plus a device suffix when there is more than
    # one, so collections from different machines never collide in `data/raw/`.
    base = arguments.output
    if base is None:
        base = blade / "paper/data/raw" / f"{collection_id}-{host_label()}"
    base = base.resolve()

    outputs = []
    for device in devices:
        output = base if len(devices) == 1 else Path(f"{base}-{device.slug}")
        collect_device(
            device=device,
            output=output,
            blade=blade,
            wgpu=wgpu,
            blade_binary=blade_binary,
            wgpu_binary=wgpu_binary,
            collection_id=collection_id,
            repository_metadata=repository_metadata,
            seed=seed,
            arguments=arguments,
        )
        outputs.append(output)

    print(f"Collected {len(outputs)} device(s):", file=sys.stderr)
    for output in outputs:
        print(f"  {output}", file=sys.stderr)


if __name__ == "__main__":
    main()

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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKLOADS = (
    "compute-independent",
    "compute-chain",
    "graphics-independent",
    "graphics-chain",
)


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
    parser.add_argument("--elements", type=int, default=1 << 20)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--warmups", type=int, default=10)
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


def parse_metadata(output: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in output.splitlines():
        if not line.startswith("#"):
            continue
        fields = next(csv.reader([line[1:].strip()]))
        if len(fields) >= 2:
            metadata[fields[0].strip()] = ",".join(fields[1:]).strip()
    return metadata


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


def main() -> None:
    arguments = parse_arguments()
    ensure_positive(arguments)
    blade = arguments.blade.resolve()
    wgpu = arguments.wgpu.resolve()
    collection_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        arguments.output
        if arguments.output is not None
        else blade / "paper/data/raw" / collection_id
    ).resolve()
    if output.exists():
        raise ValueError(f"refusing to reuse output directory: {output}")
    if not (blade / "Cargo.toml").is_file():
        raise ValueError(f"not a Blade checkout: {blade}")
    if not (wgpu / "examples/standalone/sync_bench/Cargo.toml").is_file():
        raise ValueError(f"wgpu sync benchmark is missing from: {wgpu}")

    repositories = {"blade": blade, "wgpu": wgpu}
    repository_metadata: dict[str, dict[str, str]] = {}
    for name, root in repositories.items():
        status = git_output(root, "status", "--porcelain")
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
    (output / "workload-shaders.txt").write_text(
        equivalence.stdout, encoding="utf-8"
    )

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
    manifest: dict[str, object] = {
        "schema": "blade-sync-study-v1",
        "collection_id": collection_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "backend": arguments.backend,
        "seed": seed,
        "repositories": repository_metadata,
        "parameters": {
            "repetitions": arguments.repetitions,
            "passes": arguments.passes,
            "elements": arguments.elements,
            "rounds": arguments.rounds,
            "width": arguments.width,
            "height": arguments.height,
            "warmups": arguments.warmups,
            "samples": arguments.samples,
            "validation": arguments.validation,
            "gpu_timing": not arguments.cpu_only,
            "allow_software": arguments.allow_software,
            "blade_device_id": arguments.blade_device_id,
            "wgpu_adapter_name": arguments.wgpu_adapter_name,
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
            write_command_capture(output, "vulkaninfo.txt", ["vulkaninfo", "--summary"], blade)
            write_command_capture(
                output, "vulkaninfo-full.txt", ["vulkaninfo"], blade, timeout=60
            )
            write_command_capture(output, "nvidia-smi.txt", ["nvidia-smi", "-q"], blade)
            write_command_capture(output, "rocm-smi.txt", ["rocm-smi", "--showallinfo"], blade)
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

    blade_policies = (
        ("automatic", "hazard-only", "explicit-all")
        if arguments.backend == "vulkan"
        else ("automatic",)
    )
    configurations = [
        ("blade", workload, policy)
        for workload in WORKLOADS
        for policy in blade_policies
    ]
    configurations.extend(("wgpu", workload, "tracked") for workload in WORKLOADS)
    rng = random.Random(seed)
    common_arguments = [
        "--passes",
        str(arguments.passes),
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
    validation_hashes: dict[tuple[int, str], set[str]] = {}
    with order_path.open("w", encoding="utf-8", newline="") as order_file:
        order_writer = csv.writer(order_file)
        order_writer.writerow(
            ("repetition", "index", "implementation", "workload", "policy", "file")
        )
        for repetition in range(1, arguments.repetitions + 1):
            order = list(configurations)
            rng.shuffle(order)
            for index, (implementation, workload, policy) in enumerate(order):
                run_id = (
                    f"r{repetition:02d}__{implementation}__{workload}__{policy}"
                )
                csv_name = f"{run_id}.csv"
                command = [
                    str(blade_binary if implementation == "blade" else wgpu_binary),
                    "--workload",
                    workload,
                    "--policy",
                    policy,
                    *common_arguments,
                ]
                environment = os.environ.copy()
                relevant_environment: dict[str, str] = {}
                if implementation == "blade" and arguments.blade_device_id is not None:
                    command.extend(
                        ["--device-id", str(arguments.blade_device_id)]
                    )
                if implementation == "wgpu":
                    environment["WGPU_BACKEND"] = arguments.backend
                    relevant_environment["WGPU_BACKEND"] = arguments.backend
                    if arguments.wgpu_adapter_name:
                        environment["WGPU_ADAPTER_NAME"] = arguments.wgpu_adapter_name
                        relevant_environment["WGPU_ADAPTER_NAME"] = (
                            arguments.wgpu_adapter_name
                        )

                print(run_id, file=sys.stderr, flush=True)
                result = run(command, blade, env=environment, timeout=900, check=False)
                if result.stderr:
                    (output / f"{run_id}.stderr.txt").write_text(
                        result.stderr, encoding="utf-8"
                    )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"{run_id} failed with exit code {result.returncode}\n"
                        f"{result.stdout}{result.stderr}"
                    )
                captured = (
                    f"# collection_id,{collection_id}\n"
                    f"# repetition,{repetition}\n"
                    f"{result.stdout}"
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
                validation_hashes.setdefault((repetition, workload), set()).add(
                    validation_hash
                )
                order_writer.writerow(
                    (
                        repetition,
                        index,
                        implementation,
                        workload,
                        policy,
                        csv_name,
                    )
                )
                manifest["runs"].append(
                    {
                        "id": run_id,
                        "file": csv_name,
                        "command": command,
                        "environment": relevant_environment,
                        "metadata": metadata,
                    }
                )

    for (repetition, workload), hashes in validation_hashes.items():
        if len(hashes) != 1:
            raise ValueError(
                f"repetition {repetition} workload {workload} produced "
                f"different validation hashes: {sorted(hashes)}"
            )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Raw results: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()

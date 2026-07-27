#!/usr/bin/env python3
"""Collect a diagnostic whole-process CPU profile of the matched benchmarks.

The study reports a host-time gap but does not attribute it to resource
tracking without interval-specific evidence. This script groups flat `perf`
self time by crate or library. Its current benchmark still waits for GPU
completion after every iteration, so the result is a whole-process diagnostic,
not an attribution of the record-and-submit interval. A gated or batched
profile is required for that stronger claim.

Tiny dispatches, small targets, many passes, and no timestamp queries increase
the relative amount of host work, but they did not eliminate completion waits
in the retained profiles.

Requires `perf` and a permissive `kernel.perf_event_paranoid`; the script says
so and stops rather than producing a misleading empty profile.

Usage:
    python3 paper/profile-hosts.py --wgpu ../wgpu
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# Ordered: the first pattern that matches a (dso, symbol) pair wins, so put the
# specific buckets above the general ones.
BUCKETS: tuple[tuple[str, str], ...] = (
    # `init_tracker` is lazy zero-initialisation, not resource state tracking.
    # It must be matched first, or it inflates the number this profile exists
    # to measure.
    ("wgpu init tracker", r"init_tracker|InitTracker"),
    ("wgpu tracker", r"wgpu_core\d*::track|wgpu_core\d*::.*[Tt]racker|ResourceTracker"),
    ("wgpu command", r"wgpu_core\d*::command"),
    ("wgpu validation", r"wgpu_core\d*::(validation|binding_model|pipeline)"),
    ("wgpu device/resource", r"wgpu_core\d*::(device|resource|storage|registry|hub)"),
    ("wgpu other", r"wgpu_core\d*::|wgpu\d*::|wgt::"),
    ("wgpu-hal", r"wgpu_hal"),
    ("blade", r"blade_graphics|blade_macros|sync_bench"),
    # Before the loader bucket, and deliberately so: a Mesa ICD is named
    # `libvulkan_radeon.so` or `libvulkan_intel.so`, which a `libvulkan`
    # pattern also matches. With the loader first, every Mesa driver sample
    # was filed as loader overhead -- 76% of one column on an RADV machine.
    (
        "driver",
        r"libnvidia|libGLX|nvidia|radeonsi|libvulkan_|vulkan_radeon|vulkan_intel"
        r"|libdrm|iris_dri|libVkLayer",
    ),
    # The ICD loader proper is `libvulkan.so.1`; `ash` is the Rust binding
    # both implementations call through. The pattern is matched against
    # "<dso> <symbol>", so `ash::` needs a word boundary rather than `^`.
    ("ash / loader", r"\bash::|libvulkan\.so"),
    ("allocator", r"malloc|calloc|realloc|\bfree\b|_int_malloc|_int_free|arena|tcache|operator new"),
    ("libc / runtime", r"libc\.so|ld-linux|libstdc\+\+|libgcc|__memmove|__memcpy|__memset"),
    ("kernel", r"^\[kernel|^\[unknown\]$|vmlinux"),
)

# Intended to emphasize command recording. The retained flat profiles show
# that completion waits and driver activity still dominate the whole process.
HOST_SHAPED_WORKLOAD = (
    "--elements", "4096",
    "--rounds", "1",
    "--width", "64",
    "--height", "64",
    "--passes", "64",
    "--warmups", "20",
    "--no-gpu-timing",
)


def parse_arguments() -> argparse.Namespace:
    blade_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blade", type=Path, default=blade_root)
    parser.add_argument("--wgpu", type=Path, default=blade_root.parent / "wgpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument(
        "--workloads",
        default="compute-independent,graphics-independent",
        help="comma-separated; each is profiled for both implementations",
    )
    parser.add_argument("--policy", default="automatic")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--frequency", type=int, default=4999)
    parser.add_argument("--blade-device-id", type=lambda v: int(v, 0))
    parser.add_argument("--wgpu-adapter-name")
    parser.add_argument(
        "--percent-limit",
        type=float,
        default=0.0,
        help="drop symbols below this self-time percentage (default: keep all)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="profile the binaries as they are, without rebuilding with symbols",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the bucket patterns against known symbol shapes and exit",
    )
    parser.add_argument(
        "--reclassify",
        type=Path,
        metavar="DIR",
        help=(
            "rebuild the CSVs of an existing profile from its retained perf "
            "reports, without recording anything. Use after changing BUCKETS, "
            "including for collections made on machines that are not this one."
        ),
    )
    return parser.parse_args()


def check_perf() -> None:
    if shutil.which("perf") is None:
        raise SystemExit("perf is not installed; install linux-tools for this kernel")
    paranoid_path = Path("/proc/sys/kernel/perf_event_paranoid")
    if not paranoid_path.exists():
        return
    level = int(paranoid_path.read_text().strip())
    if level > 2:
        raise SystemExit(
            f"kernel.perf_event_paranoid is {level}, which denies unprivileged "
            "profiling.\nRun one of:\n"
            "  sudo sysctl kernel.perf_event_paranoid=1   # until reboot\n"
            "  sudo perf record ...                       # per invocation\n"
            "and rerun this script. Refusing to produce an empty profile."
        )


def classify(dso: str, symbol: str, binding: str = ".") -> str:
    # perf marks kernel samples with a `[k]` binding. Trust that over the
    # symbol name: `kptr_restrict` hides kernel symbols from unprivileged
    # users, so they arrive as bare addresses in an `[unknown]` object and no
    # name-based rule can catch them.
    if binding == "k":
        return "kernel"
    subject = f"{dso} {symbol}"
    for name, pattern in BUCKETS:
        if re.search(pattern, subject):
            return name
    return "other"


# Cases that have been got wrong at least once. `BUCKETS` is ordered, so any
# reordering can silently move a large share of one machine's samples into the
# wrong row; `--self-test` is how that gets caught before the paper quotes it.
CLASSIFICATION_CASES: tuple[tuple[str, str, str, str], ...] = (
    # A Mesa ICD is a driver, not the loader, and its name contains the
    # loader's. This misfiled 76% of one column on an RADV machine.
    ("libvulkan_radeon.so", "radv_CmdPipelineBarrier2", ".", "driver"),
    ("libvulkan_intel.so", "anv_QueueSubmit", ".", "driver"),
    ("libvulkan.so.1", "vkQueueSubmit", ".", "ash / loader"),
    ("sync-bench", "ash::vk::cmd_pipeline_barrier", ".", "ash / loader"),
    ("libnvidia-glcore.so.580.95", "[unknown]", ".", "driver"),
    # `kptr_restrict` hides kernel symbols, so they arrive as bare addresses
    # in an `[unknown]` object and only the binding identifies them.
    ("[unknown]", "0xffffffffa610c0c9", "k", "kernel"),
    # The same shape without the kernel binding is an unresolved userspace
    # address, which is genuinely unattributable rather than kernel time.
    ("[unknown]", "0x00007f2c1a4b2d10", ".", "other"),
    # Lazy zero-initialisation, not resource state tracking. Matching this
    # after the tracker pattern inflates the number the profile exists for.
    ("wgpu-sync-bench", "wgpu_core::init_tracker::InitTracker::check", ".", "wgpu init tracker"),
    ("wgpu-sync-bench", "wgpu_core::track::buffer::BufferTracker::set_single", ".", "wgpu tracker"),
    ("wgpu-sync-bench", "wgpu_core::command::compute::run", ".", "wgpu command"),
    ("wgpu-sync-bench", "wgpu_core::device::queue::submit", ".", "wgpu device/resource"),
    ("wgpu-sync-bench", "wgpu_hal::vulkan::command::transition", ".", "wgpu-hal"),
    ("sync-bench", "blade_graphics::vulkan::command::barrier", ".", "blade"),
    ("libc.so.6", "_int_malloc", ".", "allocator"),
    ("libc.so.6", "__memmove_avx_unaligned_erms", ".", "libc / runtime"),
    ("libexpat.so.1.11.2", "XML_ParseBuffer", ".", "other"),
)


def self_test() -> int:
    failures = 0
    for dso, symbol, binding, expected in CLASSIFICATION_CASES:
        actual = classify(dso, symbol, binding)
        if actual != expected:
            failures += 1
            print(
                f"FAIL {dso} {symbol}\n  expected {expected!r}, got {actual!r}",
                file=sys.stderr,
            )
    print(
        f"{len(CLASSIFICATION_CASES) - failures}/{len(CLASSIFICATION_CASES)} "
        "classification cases pass",
        file=sys.stderr,
    )
    return 1 if failures else 0


REPORT_LINE = re.compile(
    r"^\s*(?P<percent>\d+\.\d+)%\s+(?P<dso>\S+)\s+"
    r"(?:\[(?P<binding>[.kgutH])\]\s*)?(?P<symbol>.+?)\s*$"
)
EVENT_HEADER = re.compile(
    r"^#\s*Samples:\s.*of event '(?P<event>[^']+)'", re.MULTILINE
)
EVENT_COUNT = re.compile(
    r"^#\s*Event count \(approx\.\):\s*(?P<count>\d+)", re.MULTILINE
)


def total_cpu_ms(text: str) -> float:
    """Total CPU time in the report.

    `task-clock` counts nanoseconds, so the event count is the process's CPU
    time directly. Percentages alone cannot be compared between two programs
    that ran for different lengths of time, which is the whole point here.
    """
    match = EVENT_COUNT.search(text)
    return float(match.group("count")) / 1e6 if match else 0.0


def parse_report(text: str) -> list[tuple[float, str, str, str]]:
    """Rows of (self percent, dso, symbol, binding) from `perf report --stdio`.

    Raises if the report contains more than one event. A hybrid CPU records
    `cpu_atom` and `cpu_core` separately and `perf report` gives each its own
    100% scale, so summing across them silently produces totals above 100%.
    """
    events = EVENT_HEADER.findall(text)
    if len(set(events)) > 1:
        raise RuntimeError(
            "perf report contains several events "
            f"({', '.join(sorted(set(events)))}); their percentages are on "
            "separate scales and must not be combined. Record a single event."
        )
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = REPORT_LINE.match(line)
        if not match:
            continue
        rows.append(
            (
                float(match.group("percent")),
                match.group("dso"),
                match.group("symbol"),
                match.group("binding") or ".",
            )
        )
    return rows


def device_name(stdout: str) -> str:
    """The adapter the benchmark actually selected, from its CSV header.

    The profiling step does not pin an adapter, so on a machine with more than
    one GPU the two implementations can pick differently. Recording what each
    chose is what makes that visible instead of assumed.
    """
    for line in stdout.splitlines():
        if line.startswith("# device_name,"):
            return line.split(",", 1)[1].strip().strip('"')
    return ""


def run(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, env=env, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed with {result.returncode}\n"
            f"{result.stdout}{result.stderr}"
        )
    return result.stdout


def profile_one(
    *,
    binary: Path,
    label: str,
    workload: str,
    policy: str,
    arguments: argparse.Namespace,
    output: Path,
    environment: dict[str, str],
) -> list[tuple[float, str, str]]:
    data = output / f"{label}__{workload}.perf.data"
    command = [
        str(binary),
        "--workload",
        workload,
        "--policy",
        policy,
        "--samples",
        str(arguments.samples),
        *HOST_SHAPED_WORKLOAD,
    ]
    if label == "blade" and arguments.blade_device_id is not None:
        command += ["--device-id", str(arguments.blade_device_id)]

    print(f"profiling {label} / {workload}", file=sys.stderr, flush=True)
    stdout = run(
        [
            "perf",
            "record",
            "--quiet",
            # A software event, so a hybrid CPU reports one scale rather than
            # one per core type, and so the unit is CPU time rather than cycles
            # that mean different things on P and E cores.
            "-e",
            "task-clock",
            "-F",
            str(arguments.frequency),
            "-o",
            str(data),
            "--",
            *command,
        ],
        arguments.blade,
        env=environment,
    )
    report = run(
        [
            "perf",
            "report",
            "--stdio",
            "--no-children",
            "-g",
            "none",
            "--percent-limit",
            str(arguments.percent_limit),
            "-F",
            "overhead,dso,symbol",
            "-i",
            str(data),
        ],
        arguments.blade,
    )
    (output / f"{label}__{workload}.report.txt").write_text(report, encoding="utf-8")
    return parse_report(report), total_cpu_ms(report), device_name(stdout)


SYMBOL_FIELDS = (
    "implementation",
    "workload",
    "bucket",
    "dso",
    "symbol",
    "self_percent",
    "self_ms",
)


def bucket_rows(
    samples: Iterable[tuple[float, str, str, str]],
    implementation: str,
    workload: str,
    cpu_ms: float,
) -> list[dict[str, object]]:
    return [
        {
            "implementation": implementation,
            "workload": workload,
            "bucket": classify(dso, symbol, binding),
            "dso": dso,
            "symbol": symbol,
            "self_percent": percent,
            "self_ms": round(percent / 100.0 * cpu_ms, 4),
        }
        for percent, dso, symbol, binding in samples
    ]


def write_tables(
    output: Path,
    rows: list[dict[str, object]],
    totals_cpu_ms: dict[tuple[str, str], float],
) -> None:
    """Write `symbols.csv` and the `buckets.csv` the paper's table reads."""
    with (output / "symbols.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SYMBOL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    totals: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["implementation"], row["workload"], row["bucket"])
        entry = totals.setdefault(key, [0.0, 0.0])
        entry[0] += float(row["self_percent"])
        entry[1] += float(row["self_ms"])
    with (output / "buckets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("implementation", "workload", "bucket", "self_percent", "self_ms")
        )
        for key in sorted(totals, key=lambda k: (k[0], k[1], -totals[k][1])):
            percent, milliseconds = totals[key]
            writer.writerow((*key, f"{percent:.2f}", f"{milliseconds:.2f}"))
        for (label, workload), milliseconds in sorted(totals_cpu_ms.items()):
            writer.writerow(
                (
                    label,
                    workload,
                    "TOTAL (process CPU time)",
                    "100.00",
                    f"{milliseconds:.2f}",
                )
            )


def reclassify(directory: Path) -> None:
    """Rebuild a collection's CSVs from its retained `perf report` output.

    `perf record` writes a binary `perf.data` that only resolves symbols
    against the libraries of the machine that produced it, but the text report
    is retained beside it and carries everything the buckets are derived from.
    So a classification bug can be repaired for every machine from here,
    without asking anyone to re-profile.
    """
    reports = sorted(directory.glob("*.report.txt"))
    if not reports:
        raise SystemExit(f"no perf reports under {directory}")
    rows: list[dict[str, object]] = []
    totals_cpu_ms: dict[tuple[str, str], float] = {}
    for report in reports:
        label, _, workload = report.name.removesuffix(".report.txt").partition("__")
        if not workload:
            raise SystemExit(
                f"cannot read an implementation and workload from {report.name}"
            )
        text = report.read_text(encoding="utf-8", errors="replace")
        cpu_ms = total_cpu_ms(text)
        rows.extend(bucket_rows(parse_report(text), label, workload, cpu_ms))
        totals_cpu_ms[label, workload] = cpu_ms
    write_tables(directory, rows, totals_cpu_ms)
    print(f"Reclassified {len(reports)} reports in {directory}", file=sys.stderr)


def main() -> None:
    arguments = parse_arguments()
    if arguments.self_test:
        raise SystemExit(self_test())
    if arguments.reclassify:
        reclassify(arguments.reclassify.resolve())
        return
    check_perf()
    blade = arguments.blade.resolve()
    wgpu = arguments.wgpu.resolve()
    collection = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    output = (
        arguments.output or blade / "paper/data/raw" / f"{collection}-{host}-profile"
    ).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    if not arguments.skip_build:
        # Line tables make `perf` attribute inlined code to the right symbol
        # without changing optimisation, so the profile matches the binary the
        # timing runs used.
        build_env = os.environ | {"CARGO_PROFILE_RELEASE_DEBUG": "line-tables-only"}
        run(["cargo", "build", "--release", "--example", "sync-bench"], blade, build_env)
        run(["cargo", "build", "--release", "-p", "wgpu-sync-bench"], wgpu, build_env)

    binaries = {
        "blade": blade / "target/release/examples/sync-bench",
        "wgpu": wgpu / "target/release/wgpu-sync-bench",
    }
    for label, path in binaries.items():
        if not path.is_file():
            raise SystemExit(f"missing benchmark binary: {path}")

    rows: list[dict[str, object]] = []
    totals_cpu_ms: dict[tuple[str, str], float] = {}
    devices: dict[str, str] = {}
    for workload in arguments.workloads.split(","):
        workload = workload.strip()
        if not workload:
            continue
        for label, binary in binaries.items():
            environment = os.environ.copy()
            if label == "wgpu":
                environment["WGPU_BACKEND"] = arguments.backend
                if arguments.wgpu_adapter_name:
                    environment["WGPU_ADAPTER_NAME"] = arguments.wgpu_adapter_name
            samples, cpu_ms, device = profile_one(
                binary=binary,
                label=label,
                workload=workload,
                policy=arguments.policy if label == "blade" else "tracked",
                arguments=arguments,
                output=output,
                environment=environment,
            )
            rows.extend(bucket_rows(samples, label, workload, cpu_ms))
            totals_cpu_ms[label, workload] = cpu_ms
            devices[f"{label}/{workload}"] = device

    missing_devices = sorted(key for key, value in devices.items() if not value)
    if missing_devices:
        raise SystemExit(
            "benchmark output did not identify an adapter for:\n  "
            + "\n  ".join(missing_devices)
            + "\nNo manifest or derived tables were written."
        )
    selected = set(devices.values())
    if len(selected) > 1:
        raise SystemExit(
            "the implementations profiled different adapters:\n  "
            + "\n  ".join(f"{key} = {value}" for key, value in sorted(devices.items()))
            + "\nPass --blade-device-id and --wgpu-adapter-name to pin one "
            "physical device. No manifest or derived tables were written for "
            "this unmatched profile."
        )
    write_tables(output, rows, totals_cpu_ms)

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "blade-sync-profile-v1",
                "collection_id": collection,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "host": host,
                "platform": platform.platform(),
                "backend": arguments.backend,
                "devices": devices,
                "selectors": {
                    "blade_device_id": arguments.blade_device_id,
                    "wgpu_adapter_name": arguments.wgpu_adapter_name,
                },
                "policy": arguments.policy,
                "workloads": arguments.workloads,
                "samples": arguments.samples,
                "frequency": arguments.frequency,
                "workload_shape": list(HOST_SHAPED_WORKLOAD),
                "repositories": {
                    name: {
                        "revision": run(
                            ["git", "rev-parse", "HEAD"], root
                        ).strip(),
                        "status": run(
                            ["git", "status", "--porcelain", "--", ":(exclude)paper/data"],
                            root,
                        ).strip(),
                    }
                    for name, root in (("blade", blade), ("wgpu", wgpu))
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Profiles: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()

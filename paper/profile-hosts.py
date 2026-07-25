#!/usr/bin/env python3
"""Attribute host CPU time in the matched benchmarks to source of work.

The study reports that the matched wgpu program costs 2-12x more host time than
Blade for the same passes, and refuses to attribute that to resource tracking
without evidence. This produces the evidence: a flat `perf` profile of each
implementation, with self time grouped by the crate or library it belongs to,
so the tracker's share is a number rather than a hypothesis.

The workload is deliberately mis-shaped for the GPU and well shaped for the
host: tiny dispatches, small targets, many passes, no timestamp queries. That
keeps the process out of `wait_for` and inside the recording path, which is
what is being attributed.

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
    ("ash / loader", r"^ash::|libvulkan"),
    ("driver", r"libnvidia|libGLX|nvidia|radeonsi|vulkan_radeon|vulkan_intel|libdrm|iris_dri"),
    ("allocator", r"malloc|calloc|realloc|\bfree\b|_int_malloc|_int_free|arena|tcache|operator new"),
    ("libc / runtime", r"libc\.so|ld-linux|libstdc\+\+|libgcc|__memmove|__memcpy|__memset"),
    ("kernel", r"^\[kernel|^\[unknown\]$|vmlinux"),
)

# Small enough that the GPU finishes immediately, numerous enough that command
# recording dominates the process.
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
        default=0.05,
        help="drop symbols below this self-time percentage (default: 0.05)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="profile the binaries as they are, without rebuilding with symbols",
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


def classify(dso: str, symbol: str) -> str:
    subject = f"{dso} {symbol}"
    for name, pattern in BUCKETS:
        if re.search(pattern, subject):
            return name
    return "other"


REPORT_LINE = re.compile(
    r"^\s*(?P<percent>\d+\.\d+)%\s+(?P<dso>\S+)\s+(?P<symbol>.+?)\s*$"
)


def parse_report(text: str) -> list[tuple[float, str, str]]:
    """Rows of (self percent, dso, symbol) from `perf report --stdio`."""
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = REPORT_LINE.match(line)
        if not match:
            continue
        symbol = match.group("symbol")
        # `perf report` prefixes the symbol with its binding, e.g. "[.] name".
        symbol = re.sub(r"^\[[.k]\]\s*", "", symbol)
        rows.append((float(match.group("percent")), match.group("dso"), symbol))
    return rows


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
    run(
        [
            "perf",
            "record",
            "--quiet",
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
    return parse_report(report)


def main() -> None:
    arguments = parse_arguments()
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
            samples = profile_one(
                binary=binary,
                label=label,
                workload=workload,
                policy=arguments.policy if label == "blade" else "tracked",
                arguments=arguments,
                output=output,
                environment=environment,
            )
            for percent, dso, symbol in samples:
                rows.append(
                    {
                        "implementation": label,
                        "workload": workload,
                        "bucket": classify(dso, symbol),
                        "dso": dso,
                        "symbol": symbol,
                        "self_percent": percent,
                    }
                )

    with (output / "symbols.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "implementation",
                "workload",
                "bucket",
                "dso",
                "symbol",
                "self_percent",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    totals: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["implementation"], row["workload"], row["bucket"])
        totals[key] = totals.get(key, 0.0) + float(row["self_percent"])
    with (output / "buckets.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("implementation", "workload", "bucket", "self_percent"))
        for key in sorted(totals, key=lambda k: (k[0], k[1], -totals[k])):
            writer.writerow((*key, f"{totals[key]:.2f}"))

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "blade-sync-profile-v1",
                "collection_id": collection,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "host": host,
                "platform": platform.platform(),
                "backend": arguments.backend,
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

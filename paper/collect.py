#!/usr/bin/env python3
"""Collect everything the paper needs from one machine, in one command.

Three things are collected, in decreasing order of importance:

1. the timing matrix (`run-study-matrix.py`) --- every device, every workload,
   every policy. This is the study; without it the machine contributes nothing.
2. a host CPU profile (`profile-hosts.py`) --- needs `perf` and a permissive
   `kernel.perf_event_paranoid`.
3. RenderDoc captures of both implementations (`capture-streams.py`) --- needs
   RenderDoc, which it downloads if the machine has none.

Steps 2 and 3 are optional and independent. If a prerequisite is missing this
says which one and carries on, because a machine that can only contribute
timings should still contribute timings. Only a failure of step 1, or a crash
rather than a missing prerequisite, stops the run.

Usage:
    python3 paper/collect.py --wgpu ../wgpu
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parse_arguments() -> tuple[argparse.Namespace, list[str]]:
    blade_root = HERE.parent
    parser = argparse.ArgumentParser(
        description="Run the whole collection protocol on this machine.",
        epilog=(
            "Arguments not listed here are passed through to "
            "run-study-matrix.py, so the usual selectors still work."
        ),
    )
    parser.add_argument("--blade", type=Path, default=blade_root)
    parser.add_argument("--wgpu", type=Path, default=blade_root.parent / "wgpu")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--blade-device-id", type=lambda value: int(value, 0))
    parser.add_argument("--wgpu-adapter-name")
    # Mirror run-study-matrix.py: a macOS machine measures Metal unless told
    # otherwise. Hardcoding "vulkan" here overrode the runner's own
    # platform-aware default and sent Mac collections down the wrong backend.
    parser.add_argument(
        "--backend",
        choices=("vulkan", "metal"),
        default="metal" if sys.platform == "darwin" else "vulkan",
    )
    parser.add_argument(
        "--skip-profile", action="store_true", help="do not run profile-hosts.py"
    )
    parser.add_argument(
        "--skip-captures", action="store_true", help="do not run capture-streams.py"
    )
    parser.add_argument(
        "--sweeps",
        action="store_true",
        help="also collect the pass-count sweeps (adds roughly half an hour)",
    )
    return parser.parse_known_args()


def shared_selectors(arguments: argparse.Namespace) -> list[str]:
    selectors = ["--backend", arguments.backend]
    if arguments.blade_device_id is not None:
        selectors += ["--blade-device-id", hex(arguments.blade_device_id)]
    if arguments.wgpu_adapter_name:
        selectors += ["--wgpu-adapter-name", arguments.wgpu_adapter_name]
    return selectors


def step(
    name: str, command: list[str], cwd: Path, *, required: bool
) -> tuple[str, str]:
    """Run one collection step. Returns (status, detail) rather than raising."""
    print(f"\n=== {name}\n$ {' '.join(command)}", file=sys.stderr, flush=True)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    sys.stderr.write(result.stderr)
    if result.stdout.strip():
        sys.stderr.write(result.stdout)
    if result.returncode == 0:
        return "collected", ""
    # The optional steps exit with a message naming the missing prerequisite;
    # that is a skip, not a failure, and the operator should see the message
    # rather than a traceback. A traceback means something genuinely broke, and
    # there the useful line is the last one, not the first.
    output = (result.stderr or result.stdout).strip()
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        reason = f"exit code {result.returncode}"
    elif "Traceback (most recent call last)" in output:
        reason = lines[-1]
    else:
        reason = lines[0]
    if required:
        raise SystemExit(f"\n{name} failed, stopping:\n{reason}")
    return "skipped", reason


def main() -> None:
    arguments, passthrough = parse_arguments()
    blade = arguments.blade.resolve()
    wgpu = arguments.wgpu.resolve()
    selectors = shared_selectors(arguments)
    outcomes: list[tuple[str, str, str]] = []

    name = "timing matrix"
    status, reason = step(
        name,
        [
            sys.executable,
            str(HERE / "run-study-matrix.py"),
            "--wgpu",
            str(wgpu),
            "--repetitions",
            str(arguments.repetitions),
            *selectors,
            *passthrough,
        ],
        blade,
        required=True,
    )
    outcomes.append((name, status, reason))

    if arguments.sweeps:
        for label, extra in (
            ("pass-count sweep, GPU-timed", []),
            ("pass-count sweep, timestamp-free", ["--cpu-only"]),
        ):
            status, reason = step(
                label,
                [
                    sys.executable,
                    str(HERE / "run-study-matrix.py"),
                    "--wgpu",
                    str(wgpu),
                    "--repetitions",
                    str(arguments.repetitions),
                    "--pass-list",
                    "1,2,4,8,16,32,64",
                    *selectors,
                    *extra,
                ],
                blade,
                required=False,
            )
            outcomes.append((label, status, reason))

    if not arguments.skip_profile:
        status, reason = step(
            "host CPU profile",
            [
                sys.executable,
                str(HERE / "profile-hosts.py"),
                "--wgpu",
                str(wgpu),
                *selectors,
            ],
            blade,
            required=False,
        )
        outcomes.append(("host CPU profile", status, reason))

    if not arguments.skip_captures:
        status, reason = step(
            "command-stream captures",
            [
                sys.executable,
                str(HERE / "capture-streams.py"),
                "--wgpu",
                str(wgpu),
                *selectors,
            ],
            blade,
            required=False,
        )
        outcomes.append(("command-stream captures", status, reason))

    print("\n=== summary", file=sys.stderr)
    for name, status, reason in outcomes:
        line = f"  {status:10} {name}"
        if reason:
            line += f"\n             {reason}"
        print(line, file=sys.stderr)
    skipped = [name for name, status, _ in outcomes if status == "skipped"]
    if skipped:
        print(
            "\nThe timing matrix is what the study needs; the skipped steps add "
            "explanation, not results. Copy the whole paper/data/raw directory "
            "back either way.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Capture the Vulkan command stream of each configuration with RenderDoc.

The paper describes the barriers Blade emits from its source. This captures
what actually reaches the driver, so the description can be checked rather than
trusted, which is the open result gate in experiments.md.

The benchmark is headless, so there are no swapchain presents for RenderDoc to
delimit a capture at. `sync-bench --capture` therefore calls the in-application
API around one warmed iteration; this script arranges for the library to be
loaded, which is the only reason it needs to exist.

Requires the `renderdoc` package (Debian/Ubuntu: `apt install renderdoc`).

Usage:
    python3 paper/capture-streams.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LIBRARY_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/librenderdoc.so",
    "/usr/lib/librenderdoc.so",
    "/usr/local/lib/librenderdoc.so",
    "/opt/renderdoc/lib/librenderdoc.so",
)

# One capture per configuration is enough to read counts, scopes and layouts;
# these are not timing runs, so the shape is chosen to be small and legible in
# the RenderDoc event browser rather than representative.
CAPTURE_SHAPE = (
    "--elements", "65536",
    "--rounds", "1",
    "--width", "256",
    "--height", "256",
    "--passes", "4",
    "--warmups", "2",
    "--samples", "2",
    "--no-gpu-timing",
)

WORKLOADS = (
    "compute-independent",
    "compute-chain",
    "graphics-independent",
    "graphics-chain",
    "mixed-independent",
    "mixed-chain",
)
POLICIES = ("automatic", "automatic-scoped", "hazard-only")


def parse_arguments() -> argparse.Namespace:
    blade_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blade", type=Path, default=blade_root)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path, help="path to librenderdoc.so")
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--blade-device-id", type=lambda v: int(v, 0))
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def find_library(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"no such library: {explicit}")
        return explicit
    for candidate in LIBRARY_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    found = shutil.which("renderdoccmd")
    hint = "" if found else "\nInstall it first, e.g. `sudo apt install renderdoc`."
    raise SystemExit(
        "librenderdoc.so was not found in any of:\n  "
        + "\n  ".join(LIBRARY_CANDIDATES)
        + hint
        + "\nPass --library to point at it explicitly."
    )


def main() -> None:
    arguments = parse_arguments()
    blade = arguments.blade.resolve()
    library = find_library(arguments.library)
    collection = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    output = (
        arguments.output or blade / "paper/data/raw" / f"{collection}-{host}-captures"
    ).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    if not arguments.skip_build:
        subprocess.run(
            ["cargo", "build", "--release", "--example", "sync-bench"],
            cwd=blade,
            check=True,
        )
    binary = blade / "target/release/examples/sync-bench"
    if not binary.is_file():
        raise SystemExit(f"missing benchmark binary: {binary}")

    runs = []
    for workload in (w.strip() for w in arguments.workloads.split(",") if w.strip()):
        for policy in (p.strip() for p in arguments.policies.split(",") if p.strip()):
            command = [
                str(binary),
                "--workload",
                workload,
                "--policy",
                policy,
                "--capture",
                *CAPTURE_SHAPE,
            ]
            if arguments.blade_device_id is not None:
                command += ["--device-id", str(arguments.blade_device_id)]
            environment = os.environ.copy()
            # RenderDoc has to be resident before the Vulkan loader initialises,
            # so it is preloaded rather than dlopened by the benchmark.
            preload = environment.get("LD_PRELOAD")
            environment["LD_PRELOAD"] = (
                f"{library}:{preload}" if preload else str(library)
            )
            environment["RENDERDOC_CAPTUREOPTS"] = environment.get(
                "RENDERDOC_CAPTUREOPTS", ""
            )
            print(f"capturing {workload} / {policy}", file=sys.stderr, flush=True)
            result = subprocess.run(
                command, cwd=output, env=environment, text=True, capture_output=True
            )
            (output / f"{workload}__{policy}.log").write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            if result.returncode != 0:
                raise SystemExit(
                    f"{workload}/{policy} failed with {result.returncode}\n"
                    f"{result.stdout}{result.stderr}"
                )
            if "RenderDoc is not loaded" in result.stderr:
                raise SystemExit(
                    "the benchmark ran but RenderDoc was not loaded; check that "
                    f"{library} matches this architecture and that LD_PRELOAD is "
                    "not being stripped."
                )
            runs.append({"workload": workload, "policy": policy, "command": command})

    captures = sorted(p.name for p in output.glob("*.rdc"))
    if not captures:
        raise SystemExit(
            "no .rdc files were produced. The benchmark reported no error, so "
            "RenderDoc loaded but did not write a capture; check its log in "
            f"{output}."
        )

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "blade-sync-capture-v1",
                "collection_id": collection,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "host": host,
                "platform": platform.platform(),
                "library": str(library),
                "capture_shape": list(CAPTURE_SHAPE),
                "captures": captures,
                "runs": runs,
                "note": (
                    "Blade only. The matched wgpu benchmark has no --capture "
                    "flag; adding one that calls the same RenderDoc API around "
                    "one warmed iteration would make the comparison symmetric."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Captures: {output} ({len(captures)} files)", file=sys.stderr)


if __name__ == "__main__":
    main()

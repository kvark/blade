#!/usr/bin/env python3

import argparse
import hashlib
import re
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    blade_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blade", type=Path, default=blade_root)
    parser.add_argument("--wgpu", type=Path, default=blade_root.parent / "wgpu")
    return parser.parse_args()


def normalize_wgpu(source: str) -> str:
    source = re.sub(r"^@group\(0\) @binding\(\d+\)\n", "", source, flags=re.MULTILINE)
    return source.replace("var<immediate>", "var<uniform>")


def digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def main() -> None:
    arguments = parse_arguments()
    pairs = (
        (
            arguments.blade / "examples/sync-bench/compute.wgsl",
            arguments.wgpu / "examples/standalone/sync_bench/src/compute.wgsl",
        ),
        (
            arguments.blade / "examples/sync-bench/graphics.wgsl",
            arguments.wgpu / "examples/standalone/sync_bench/src/graphics.wgsl",
        ),
    )
    for blade_path, wgpu_path in pairs:
        blade_source = blade_path.read_text(encoding="utf-8")
        wgpu_source = normalize_wgpu(wgpu_path.read_text(encoding="utf-8"))
        if blade_source != wgpu_source:
            raise ValueError(f"workload shaders differ: {blade_path} and {wgpu_path}")
        print(f"{blade_path.name},{digest(blade_source)}")


if __name__ == "__main__":
    main()

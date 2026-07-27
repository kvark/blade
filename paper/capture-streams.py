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
import csv
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


LIBRARY_CANDIDATES = (
    "/usr/lib/x86_64-linux-gnu/librenderdoc.so",
    "/usr/lib/librenderdoc.so",
    "/usr/local/lib/librenderdoc.so",
    "/opt/renderdoc/lib/librenderdoc.so",
)

# Official upstream build, used when no system RenderDoc is installed. The hash
# is of the tarball as fetched on 2026-07-25; renderdoc.org publishes no
# checksum of its own, so this pins reruns and detects corruption but cannot
# attest to the first download. The transport guarantee is HTTPS with HSTS.
RENDERDOC_VERSION = "1.45"
RENDERDOC_URL = (
    f"https://renderdoc.org/stable/{RENDERDOC_VERSION}/"
    f"renderdoc_{RENDERDOC_VERSION}.tar.gz"
)
RENDERDOC_SHA256 = "b0a7ee8ec78c4fa511eb44137380d99a748472e5fd24c877f8afcc860a172a42"
RENDERDOC_LIBRARY = f"renderdoc_{RENDERDOC_VERSION}/lib/librenderdoc.so"

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
# The mixed families exist only in Blade's benchmark.
SHARED_WORKLOADS = frozenset(WORKLOADS) - {"mixed-independent", "mixed-chain"}
# Include the identical-request control as well as the configurations discussed
# in the result figures. The 2026-07-27 captures predate this addition and must
# be refreshed before the control is claimed as captured evidence.
POLICIES = (
    "automatic",
    "automatic-scoped",
    "hazard-only",
    "hazard-only-scoped",
    "explicit-all",
    "explicit-all-scoped",
)


def parse_benchmark_metadata(output: str) -> dict[str, str]:
    """Read the benchmark's ``# key,value`` records from captured stdout."""
    metadata: dict[str, str] = {}
    for line in output.splitlines():
        if not line.startswith("#"):
            continue
        fields = next(csv.reader([line[1:].strip()]))
        if len(fields) >= 2:
            metadata[fields[0].strip()] = ",".join(fields[1:]).strip()
    return metadata


def parse_arguments() -> argparse.Namespace:
    blade_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--blade", type=Path, default=blade_root)
    parser.add_argument("--wgpu", type=Path, default=blade_root.parent / "wgpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--library", type=Path, help="path to librenderdoc.so")
    parser.add_argument(
        "--renderdoc-cache",
        type=Path,
        default=blade_root / "paper/data/tools",
        help="where to unpack a downloaded RenderDoc (default: paper/data/tools)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of fetching RenderDoc when none is installed",
    )
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--blade-device-id", type=lambda v: int(v, 0))
    parser.add_argument("--wgpu-adapter-name")
    parser.add_argument("--backend", default="vulkan")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def download_renderdoc(cache: Path) -> Path:
    """Fetch and unpack the upstream RenderDoc build into `cache`.

    This ends with a third-party shared library being preloaded into the
    benchmark process, so it says out loud what it is fetching and from where,
    and verifies the archive against a pinned hash before unpacking it.
    """
    library = cache / RENDERDOC_LIBRARY
    if library.is_file():
        return library

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"renderdoc_{RENDERDOC_VERSION}.tar.gz"
    if not archive.is_file():
        print(
            f"RenderDoc not installed; downloading {RENDERDOC_URL}\n"
            f"  into {cache}\n"
            "  this library gets preloaded into the benchmark process",
            file=sys.stderr,
            flush=True,
        )
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(RENDERDOC_URL) as response, temporary.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(archive)

    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != RENDERDOC_SHA256:
        archive.unlink()
        raise SystemExit(
            f"downloaded archive does not match the pinned hash.\n"
            f"  expected {RENDERDOC_SHA256}\n"
            f"  got      {digest.hexdigest()}\n"
            "The archive has been deleted. If upstream republished the build, "
            "update RENDERDOC_SHA256 after checking the change is expected."
        )

    with tarfile.open(archive, "r:gz") as tar:
        # `data` rejects absolute paths, traversal, and special files.
        tar.extractall(cache, filter="data")
    if not library.is_file():
        raise SystemExit(
            f"{archive} unpacked but {RENDERDOC_LIBRARY} is not in it; the "
            "archive layout has changed."
        )
    repair_layer_manifest(cache, library)
    return library


def repair_layer_manifest(cache: Path, library: Path) -> Path:
    """Rewrite the shipped layer manifest to point at the unpacked library.

    RenderDoc hooks Vulkan as a layer, not through the preloaded library alone,
    and the manifest in the tarball carries the absolute `library_path` of
    upstream's build machine. Without this the loader silently finds no layer,
    the benchmark runs happily, and no capture is written.
    """
    manifest = (
        cache / f"renderdoc_{RENDERDOC_VERSION}/etc/vulkan/implicit_layer.d"
        / "renderdoc_capture.json"
    )
    if not manifest.is_file():
        raise SystemExit(f"layer manifest missing from the archive: {manifest}")
    document = json.loads(manifest.read_text())
    document["layer"]["library_path"] = str(library)
    manifest.write_text(json.dumps(document, indent=4) + "\n", encoding="utf-8")
    return manifest.parent


def layer_environment(library: Path) -> dict[str, str]:
    """Environment that makes the Vulkan loader load RenderDoc's layer.

    A system install registers its manifest where the loader already looks; an
    unpacked tarball does not, so the directory is added explicitly. The layer
    is implicit and gated behind its own enable variable, which is set either
    way.
    """
    environment = {"ENABLE_VULKAN_RENDERDOC_CAPTURE": "1"}
    manifests = library.parent.parent / "etc/vulkan/implicit_layer.d"
    if manifests.is_dir():
        # Newer loaders take this directly; older ones look under XDG_DATA_DIRS,
        # so both are provided.
        environment["VK_ADD_IMPLICIT_LAYER_PATH"] = str(manifests)
        environment["XDG_DATA_DIRS"] = os.pathsep.join(
            filter(
                None,
                [
                    str(library.parent.parent / "etc"),
                    os.environ.get("XDG_DATA_DIRS", ""),
                ],
            )
        )
    return environment


def find_library(
    explicit: Path | None, cache: Path, allow_download: bool
) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise SystemExit(f"no such library: {explicit}")
        return explicit
    for candidate in LIBRARY_CANDIDATES:
        if Path(candidate).is_file():
            return Path(candidate)
    cached = cache / RENDERDOC_LIBRARY
    if cached.is_file():
        return cached
    if not allow_download:
        raise SystemExit(
            "librenderdoc.so was not found in any of:\n  "
            + "\n  ".join((*LIBRARY_CANDIDATES, str(cached)))
            + "\nInstall it (`sudo apt install renderdoc`), pass --library, or "
            "drop --no-download to fetch the upstream build."
        )
    return download_renderdoc(cache)


STAGE_BITS = {
    0x1: "TOP_OF_PIPE", 0x2: "DRAW_INDIRECT", 0x4: "VERTEX_INPUT",
    0x8: "VERTEX_SHADER", 0x10: "TESS_CONTROL", 0x20: "TESS_EVAL",
    0x40: "GEOMETRY", 0x80: "FRAGMENT", 0x100: "EARLY_FRAGMENT_TESTS",
    0x200: "LATE_FRAGMENT_TESTS", 0x400: "COLOR_ATTACHMENT_OUTPUT",
    0x800: "COMPUTE_SHADER", 0x1000: "TRANSFER", 0x2000: "BOTTOM_OF_PIPE",
    0x4000: "HOST", 0x8000: "ALL_GRAPHICS", 0x10000: "ALL_COMMANDS",
}
ACCESS_BITS = {
    0x1: "INDIRECT_COMMAND_READ", 0x2: "INDEX_READ", 0x4: "VERTEX_ATTRIBUTE_READ",
    0x8: "UNIFORM_READ", 0x10: "INPUT_ATTACHMENT_READ", 0x20: "SHADER_READ",
    0x40: "SHADER_WRITE", 0x80: "COLOR_ATTACHMENT_READ",
    0x100: "COLOR_ATTACHMENT_WRITE", 0x200: "DEPTH_STENCIL_ATTACHMENT_READ",
    0x400: "DEPTH_STENCIL_ATTACHMENT_WRITE", 0x800: "TRANSFER_READ",
    0x1000: "TRANSFER_WRITE", 0x2000: "HOST_READ", 0x4000: "HOST_WRITE",
    0x8000: "MEMORY_READ", 0x10000: "MEMORY_WRITE",
}
LAYOUTS = {
    0: "UNDEFINED", 1: "GENERAL", 2: "COLOR_ATTACHMENT_OPTIMAL",
    3: "DEPTH_STENCIL_ATTACHMENT_OPTIMAL", 4: "DEPTH_STENCIL_READ_ONLY_OPTIMAL",
    5: "SHADER_READ_ONLY_OPTIMAL", 6: "TRANSFER_SRC_OPTIMAL",
    7: "TRANSFER_DST_OPTIMAL", 8: "PREINITIALIZED", 1000001002: "PRESENT_SRC_KHR",
}


def decode(mask: int, names: dict[int, str]) -> str:
    if mask == 0:
        return "NONE"
    parts = [name for bit, name in sorted(names.items()) if mask & bit]
    remainder = mask & ~sum(bit for bit in names if mask & bit)
    if remainder:
        parts.append(f"0x{remainder:x}")
    return "|".join(parts)


def field(element, name: str) -> int | None:
    for child in element:
        if child.get("name") == name:
            text = (child.text or "").strip()
            if text:
                try:
                    return int(text)
                except ValueError:
                    return None
    return None


def resource_field(element, *names: str) -> str | None:
    """Return a direct ResourceId field without coercing its opaque value."""
    for child in element:
        if child.tag == "ResourceId" and child.get("name") in names:
            text = (child.text or "").strip()
            if text:
                return text
    return None


def extract_barriers(
    xml_path: Path, implementation: str, workload: str, policy: str
) -> list[dict]:
    """One row per `vkCmdPipelineBarrier` in the capture.

    This is the point of capturing: the paper describes these masks from
    Blade's source, and this is what the driver actually received. `after_work`
    counts submitted dispatch/draw commands preceding the barrier. RenderDoc's
    XML lists API recording calls in host order, but wgpu records each pass and
    transition into separate Vulkan command buffers and orders those buffers
    only in `vkQueueSubmit`. We therefore replay commands in submission order;
    counting preceding XML chunks would assign wgpu's barriers to the wrong
    pass boundaries.
    """
    import xml.etree.ElementTree as ElementTree

    root = ElementTree.parse(xml_path).getroot()
    command_events: dict[str, list[tuple[str, dict | None]]] = {}
    submission_order: list[str] = []
    recorded_barriers = 0

    for chunk in root.iter("chunk"):
        name = chunk.get("name") or ""

        if name.startswith("vkQueueSubmit"):
            # Vulkan 1.0/1.1 submission.
            for array in chunk.iter("array"):
                if array.get("name") == "pCommandBuffers":
                    submission_order.extend(
                        (child.text or "").strip()
                        for child in array
                        if child.tag == "ResourceId" and (child.text or "").strip()
                    )
                # Vulkan 1.3 `vkQueueSubmit2`.
                elif array.get("name") == "pCommandBufferInfos":
                    for structure in array:
                        command_buffer = resource_field(structure, "commandBuffer")
                        if command_buffer is not None:
                            submission_order.append(command_buffer)

        command_buffer = resource_field(chunk, "commandBuffer")
        if command_buffer is None:
            continue
        events = command_events.setdefault(command_buffer, [])
        if name.startswith(("vkCmdDispatch", "vkCmdDraw")):
            events.append(("work", None))
            continue
        if "PipelineBarrier" not in name:
            continue

        memory = field(chunk, "memoryBarrierCount") or 0
        buffers = field(chunk, "bufferMemoryBarrierCount") or 0
        images = field(chunk, "imageMemoryBarrierCount") or 0
        source_access = destination_access = 0
        old_layout = new_layout = None
        for container in chunk:
            if container.get("name") in ("pMemoryBarriers", "pBufferMemoryBarriers"):
                for structure in container:
                    source_access |= field(structure, "srcAccessMask") or 0
                    destination_access |= field(structure, "dstAccessMask") or 0
            if container.get("name") == "pImageMemoryBarriers":
                for structure in container:
                    source_access |= field(structure, "srcAccessMask") or 0
                    destination_access |= field(structure, "dstAccessMask") or 0
                    old_layout = field(structure, "oldLayout")
                    new_layout = field(structure, "newLayout")
        events.append(
            (
                "barrier",
                {
                    "implementation": implementation,
                    "workload": workload,
                    "policy": policy,
                    "src_stage": decode(
                        field(chunk, "srcStageMask") or 0, STAGE_BITS
                    ),
                    "dst_stage": decode(
                        field(chunk, "destStageMask") or 0, STAGE_BITS
                    ),
                    "src_access": decode(source_access, ACCESS_BITS),
                    "dst_access": decode(destination_access, ACCESS_BITS),
                    "memory_barriers": memory,
                    "buffer_barriers": buffers,
                    "image_barriers": images,
                    "old_layout": LAYOUTS.get(
                        old_layout, old_layout if old_layout is not None else ""
                    ),
                    "new_layout": LAYOUTS.get(
                        new_layout, new_layout if new_layout is not None else ""
                    ),
                },
            )
        )
        recorded_barriers += 1

    rows: list[dict] = []
    work_commands = 0
    for command_buffer in submission_order:
        for event, row in command_events.get(command_buffer, ()):
            if event == "work":
                work_commands += 1
            else:
                assert row is not None
                row["index"] = len(rows)
                row["after_work"] = work_commands
                rows.append(row)

    if recorded_barriers and len(rows) != recorded_barriers:
        raise ValueError(
            f"{xml_path}: recorded {recorded_barriers} barriers but found "
            f"{len(rows)} in submitted command buffers"
        )
    return rows


def convert_and_extract(output: Path, library: Path, runs: list[dict]) -> int:
    """Convert each capture to XML and tabulate its barriers."""
    root = library.parent.parent
    command_line = root / "bin/renderdoccmd"
    if not command_line.is_file():
        command_line = Path(shutil.which("renderdoccmd") or "")
    if not command_line or not Path(command_line).is_file():
        print(
            "renderdoccmd not found; captures are written but not tabulated",
            file=sys.stderr,
        )
        return 0
    environment = os.environ | {"LD_LIBRARY_PATH": str(root / "lib")}
    rows: list[dict] = []
    for capture in sorted(output.glob("*.rdc")):
        stem = capture.stem.removesuffix("_capture")
        implementation = "wgpu" if stem.startswith("wgpu-sync-bench__") else "blade"
        stem = stem.removeprefix("wgpu-sync-bench__").removeprefix("sync-bench__")
        workload, _, policy = stem.rpartition("__")
        xml_path = capture.with_suffix(".xml")
        result = subprocess.run(
            [str(command_line), "convert", "-f", str(capture), "-o", str(xml_path),
             "-c", "xml"],
            env=environment, text=True, capture_output=True,
        )
        if result.returncode != 0 or not xml_path.is_file():
            print(f"could not convert {capture.name}", file=sys.stderr)
            continue
        rows.extend(extract_barriers(xml_path, implementation, workload, policy))
        xml_path.unlink()
    if not rows:
        return 0
    with (output / "barriers.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    arguments = parse_arguments()
    blade = arguments.blade.resolve()
    library = find_library(
        arguments.library,
        arguments.renderdoc_cache.resolve(),
        not arguments.no_download,
    )
    collection = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    host = socket.gethostname().split(".")[0]
    output = (
        arguments.output or blade / "paper/data/raw" / f"{collection}-{host}-captures"
    ).resolve()
    if output.exists():
        raise SystemExit(f"refusing to reuse output directory: {output}")
    output.mkdir(parents=True)

    wgpu = arguments.wgpu.resolve()
    if not arguments.skip_build:
        subprocess.run(
            ["cargo", "build", "--release", "--example", "sync-bench"],
            cwd=blade,
            check=True,
        )
        subprocess.run(
            ["cargo", "build", "--release", "-p", "wgpu-sync-bench"],
            cwd=wgpu,
            check=True,
        )
    binaries = {
        "blade": blade / "target/release/examples/sync-bench",
        "wgpu": wgpu / "target/release/wgpu-sync-bench",
    }
    for label, path in binaries.items():
        if not path.is_file():
            raise SystemExit(f"missing benchmark binary: {path}")

    # wgpu has one configuration; Blade has one per policy. Pair them so both
    # command streams come from the same workload and the same machine state.
    configurations = [
        ("blade", workload, policy)
        for workload in (w.strip() for w in arguments.workloads.split(",") if w.strip())
        for policy in (p.strip() for p in arguments.policies.split(",") if p.strip())
    ]
    configurations += [
        ("wgpu", workload, "tracked")
        for workload in (w.strip() for w in arguments.workloads.split(",") if w.strip())
        if workload in SHARED_WORKLOADS
    ]

    runs = []
    device_names: dict[str, set[str]] = {}
    validation_hashes: dict[str, set[str]] = {}
    for implementation, workload, policy in configurations:
        binary = binaries[implementation]
        command = [
            str(binary),
            "--workload",
            workload,
            "--policy",
            policy,
            "--capture",
            *CAPTURE_SHAPE,
        ]
        if implementation == "blade":
            if arguments.blade_device_id is not None:
                command += ["--device-id", str(arguments.blade_device_id)]
        else:
            # wgpu takes no --policy; its single configuration is implied.
            index = command.index("--policy")
            del command[index : index + 2]
        environment = os.environ.copy()
        relevant_environment: dict[str, str] = {}
        if implementation == "wgpu":
            environment["WGPU_BACKEND"] = arguments.backend
            relevant_environment["WGPU_BACKEND"] = arguments.backend
            if arguments.wgpu_adapter_name:
                environment["WGPU_ADAPTER_NAME"] = arguments.wgpu_adapter_name
                relevant_environment["WGPU_ADAPTER_NAME"] = (
                    arguments.wgpu_adapter_name
                )
        # RenderDoc has to be resident before the Vulkan loader initialises,
        # so it is preloaded rather than dlopened by the benchmark.
        preload = environment.get("LD_PRELOAD")
        environment["LD_PRELOAD"] = (
            f"{library}:{preload}" if preload else str(library)
        )
        environment.update(layer_environment(library))
        print(
            f"capturing {implementation} / {workload} / {policy}",
            file=sys.stderr,
            flush=True,
        )
        captures_before = set(output.glob("*.rdc"))
        result = subprocess.run(
            command, cwd=output, env=environment, text=True, capture_output=True
        )
        (output / f"{implementation}__{workload}__{policy}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            raise SystemExit(
                f"{implementation}/{workload}/{policy} failed with "
                f"{result.returncode}\n"
                f"{result.stdout}{result.stderr}"
            )
        if "RenderDoc is not loaded" in result.stderr:
            raise SystemExit(
                "the benchmark ran but RenderDoc was not loaded; check that "
                f"{library} matches this architecture and that LD_PRELOAD is "
                "not being stripped."
            )
        new_captures = sorted(
            path.name for path in set(output.glob("*.rdc")) - captures_before
        )
        if not new_captures:
            raise SystemExit(
                f"{implementation}/{workload}/{policy} produced no capture. "
                "RenderDoc's "
                "in-application API was reachable, so the library loaded, "
                "but its Vulkan layer did not intercept anything. Check "
                "that VK_ADD_IMPLICIT_LAYER_PATH reaches this loader "
                f"({environment.get('VK_ADD_IMPLICIT_LAYER_PATH')}) and "
                "that the manifest's library_path is correct."
            )
        if len(new_captures) != 1:
            raise SystemExit(
                f"{implementation}/{workload}/{policy} produced "
                f"{len(new_captures)} captures instead of one: {new_captures}"
            )
        metadata = parse_benchmark_metadata(result.stdout)
        expected_metadata = {
            "schema": "blade-sync-bench-v1",
            "implementation": implementation,
            "backend": arguments.backend,
            "gpu_timing": "false",
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise SystemExit(
                    f"{implementation}/{workload}/{policy}: metadata {key} is "
                    f"{metadata.get(key)!r}, expected {expected!r}"
                )
        device_name = metadata.get("device_name")
        if not device_name:
            raise SystemExit(
                f"{implementation}/{workload}/{policy}: missing device_name "
                "metadata"
            )
        validation_hash = metadata.get("validation_hash")
        if not validation_hash:
            raise SystemExit(
                f"{implementation}/{workload}/{policy}: missing "
                "validation_hash metadata"
            )
        device_names.setdefault(implementation, set()).add(device_name)
        validation_hashes.setdefault(workload, set()).add(validation_hash)
        runs.append(
            {
                "implementation": implementation,
                "workload": workload,
                "policy": policy,
                "command": command,
                "environment": relevant_environment,
                "metadata": metadata,
                "captures": new_captures,
            }
        )

    captures = sorted(p.name for p in output.glob("*.rdc"))
    if not captures:
        raise SystemExit(
            "no .rdc files were produced. The benchmark reported no error, so "
            "RenderDoc loaded but did not write a capture; check its log in "
            f"{output}."
        )
    if len(captures) != len(configurations):
        raise SystemExit(
            f"expected one capture for each of {len(configurations)} "
            f"configurations, found {len(captures)}"
        )
    selected_devices = {
        name
        for implementation_names in device_names.values()
        for name in implementation_names
    }
    if len(selected_devices) != 1:
        raise SystemExit(
            "implementations selected different devices: "
            + "; ".join(
                f"{implementation}={sorted(names)}"
                for implementation, names in sorted(device_names.items())
            )
            + "\nPass --blade-device-id and --wgpu-adapter-name to pin one "
            "physical device."
        )
    conflicts = {
        workload: sorted(hashes)
        for workload, hashes in validation_hashes.items()
        if len(hashes) != 1
    }
    if conflicts:
        raise SystemExit(
            "capture configurations produced different validation hashes: "
            + "; ".join(
                f"{workload}={hashes}"
                for workload, hashes in sorted(conflicts.items())
            )
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
                "library_source": (
                    "downloaded"
                    if str(library).startswith(str(arguments.renderdoc_cache.resolve()))
                    else "system"
                ),
                "capture_shape": list(CAPTURE_SHAPE),
                "captures": captures,
                "runs": runs,
                "backend": arguments.backend,
                "selected_device": next(iter(selected_devices)),
                "validation_hashes": {
                    workload: next(iter(hashes))
                    for workload, hashes in sorted(validation_hashes.items())
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    barriers = convert_and_extract(output, library, runs)
    print(
        f"Captures: {output} ({len(captures)} files, {barriers} barriers tabulated)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

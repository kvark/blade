#!/usr/bin/env python3
"""Generate the LaTeX tables used by main.tex from the raw collections.

Only the Python standard library is used, matching `analyze.py`. Every number
printed by the paper comes from this script, so a table can never drift from
the measurements it claims to summarize.

Usage:
    python3 paper/build-tables.py \
        --raw paper/data/raw \
        --output paper/data/derived/tables
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent


def load_analyze():
    spec = importlib.util.spec_from_file_location("analyze", HERE / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `analyze` defines a dataclass, which resolves annotations through
    # `sys.modules`; register the module before executing it.
    sys.modules["analyze"] = module
    spec.loader.exec_module(module)
    return module


analyze = load_analyze()


# Collections whose label or role cannot be inferred from their contents, and
# the order in which the paper presents the fixed matrix. Anything else found
# under `data/raw/` is discovered and classified automatically, so a retest on
# a new machine needs no edit here.
KNOWN_COLLECTIONS: dict[str, tuple[str, str]] = {}
MATRIX_ORDER = ("zork", "rubik", "k6", "matrix", "mac")

DEVICE_SHORT = {
    "NVIDIA GeForce RTX 5070": "RTX 5070",
    "AMD Radeon RX 7900 XT (RADV NAVI31)": "RX 7900 XT",
    "AMD Ryzen 5 9600X 6-Core Processor (RADV RAPHAEL_MENDOCINO)": "Raphael iGPU",
    "AMD Radeon 780M Graphics (RADV PHOENIX)": "Radeon 780M",
    "Intel(R) Graphics (RPL-U)": "Intel Xe (RPL-U)",
    "Apple M3": "Apple M3",
}

WORKLOAD_SHORT = {
    "compute-independent": "c-ind",
    "compute-chain": "c-chain",
    "graphics-independent": "g-ind",
    "graphics-chain": "g-chain",
    "mixed-independent": "m-ind",
    "mixed-chain": "m-chain",
}

WORKLOAD_ORDER = (
    "compute-independent",
    "compute-chain",
    "graphics-independent",
    "graphics-chain",
    "mixed-independent",
    "mixed-chain",
)

BOOTSTRAP_SAMPLES = 10_000

# Sentinel for a row that carries a bare booktabs rule rather than cells.
RULE = "\\addlinespace"

DRIVER_SHORT = {
    "Intel open-source Mesa driver": "anv",
    "NVIDIA": "NVIDIA",
    "radv": "radv",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=HERE / "data/raw")
    parser.add_argument("--output", type=Path, default=HERE / "data/derived/tables")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    return parser.parse_args()


def discover_collections(raw: Path) -> list[Collection]:
    """Every collection under `raw`, classified by what it actually contains.

    A collection that varies the pass count is a sweep; anything else is a
    fixed matrix. Machines are ordered by `MATRIX_ORDER` first, then
    alphabetically, so a newly collected machine appears at the end rather than
    reshuffling the paper.
    """
    collections = []
    for root in sorted(raw.iterdir()):
        if not (root / "manifest.json").is_file():
            continue
        label, role = KNOWN_COLLECTIONS.get(root.name, (None, None))
        collections.append(Collection(root, label, role))

    def sort_key(collection: Collection) -> tuple:
        try:
            rank = MATRIX_ORDER.index(collection.label)
        except ValueError:
            rank = len(MATRIX_ORDER)
        return (rank, collection.device_name, collection.root.name)

    return sorted(collections, key=sort_key)


def newest_per_device(collections: list[Collection]) -> list[Collection]:
    """One matrix collection per machine and device: the most recent.

    A retest supersedes an earlier run of the same hardware rather than
    appearing beside it, so the tables never show one device twice.
    """
    best: dict[tuple[str, str], Collection] = {}
    for collection in collections:
        if collection.role != "matrix":
            continue
        key = (collection.label, collection.device_name)
        previous = best.get(key)
        if previous is None or collection.collected_utc > previous.collected_utc:
            best[key] = collection
    return [c for c in collections if best.get((c.label, c.device_name)) is c]


class Collection:
    def __init__(self, root: Path, label: str | None, role: str | None) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.label = label or self._infer_label()
        self.role = role  # finalized after the samples are loaded
        self.samples: dict[tuple, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.devices: dict[str, str] = {}
        for path in sorted(root.glob("r*.csv")):
            run = analyze.read_run(path)
            implementation = run.metadata["implementation"]
            self.devices[implementation] = run.metadata["device_name"]
            for row in run.rows:
                key = (
                    implementation,
                    row["policy"],
                    row["workload"],
                    int(row["passes"]),
                )
                bucket = self.samples[key]
                for metric in analyze.METRICS:
                    bucket[metric].append(float(row[metric]))
                bucket["host_ns"].append(
                    float(row["record_ns"]) + float(row["submit_ns"])
                )
        if self.role is None:
            self.role = self._infer_role()

    def _infer_label(self) -> str:
        host = self.manifest.get("host")
        if host:
            return str(host).split(".")[0]
        # Fall back to the trailing segment of `<timestamp>-<host>[-<device>]`.
        parts = self.root.name.split("-", 1)
        return parts[1] if len(parts) > 1 else self.root.name

    def _infer_role(self) -> str:
        if len(self.pass_counts()) > 1:
            gpu_timing = self.manifest.get("parameters", {}).get("gpu_timing", True)
            return "sweep-gpu" if gpu_timing else "sweep-cpu"
        return "matrix"

    @property
    def has_scope_axis(self) -> bool:
        return any(key[1] == "automatic-scoped" for key in self.samples)

    @property
    def collected_utc(self) -> str:
        return str(self.manifest.get("created_utc", ""))

    @property
    def blade_revision(self) -> str:
        repositories = self.manifest.get("repositories", {})
        return str(repositories.get("blade", {}).get("revision", ""))[:7]

    @property
    def is_dirty(self) -> bool:
        """Whether either repository had uncommitted changes when collected."""
        return any(
            repository.get("status")
            for repository in self.manifest.get("repositories", {}).values()
        )

    @property
    def device_name(self) -> str:
        names = sorted(set(self.devices.values()))
        return names[0] if len(names) == 1 else " / ".join(names)

    @property
    def mismatched_devices(self) -> bool:
        return len(set(self.devices.values())) > 1

    def values(
        self, implementation: str, policy: str, workload: str, passes: int, metric: str
    ) -> list[float]:
        return self.samples.get((implementation, policy, workload, passes), {}).get(
            metric, []
        )

    def pass_counts(self) -> list[int]:
        return sorted({key[3] for key in self.samples})


def median_us(values: Sequence[float]) -> float:
    return statistics.median(values) / 1000.0


def relative_difference(
    baseline: Sequence[float], contender: Sequence[float], bootstrap_samples: int, seed: Sequence[str]
) -> tuple[float, float, float]:
    """Return percent difference of medians with a bootstrap 95% interval."""
    baseline_median = statistics.median(baseline)
    contender_median = statistics.median(contender)
    low, high = analyze.bootstrap_difference_interval(
        baseline, contender, bootstrap_samples, seed
    )
    scale = 100.0 / baseline_median if baseline_median else float("nan")
    return (
        (contender_median - baseline_median) * scale,
        low * scale,
        high * scale,
    )


def signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}"


def interval(low: float, high: float, digits: int = 1) -> str:
    return f"[{low:+.{digits}f},\\,{high:+.{digits}f}]"


def latex_table(
    caption: str,
    label: str,
    column_spec: str,
    header: Sequence[str],
    body: Iterable[Sequence[str]],
    *,
    star: bool = False,
    note: str | None = None,
) -> str:
    environment = "table*" if star else "table"
    lines = [
        f"% Generated by paper/build-tables.py -- do not edit by hand.",
        f"\\begin{{{environment}}}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        "\\midrule",
    ]
    for row in body:
        # A single-element row holding a bare rule command is emitted verbatim,
        # without column separators or a row terminator.
        if len(row) == 1 and row[0] == RULE:
            lines.append("\\addlinespace")
        else:
            lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    if note:
        lines.append(f"\\\\[2pt]\\footnotesize {note}")
    lines.extend(
        [f"\\caption{{{caption}}}", f"\\label{{{label}}}", f"\\end{{{environment}}}", ""]
    )
    return "\n".join(lines)


def vulkan_facts(root: Path) -> tuple[str, str]:
    """Return (API version, unified-image-layouts support) from vulkaninfo output."""
    summary = root / "vulkaninfo.txt"
    full = root / "vulkaninfo-full.txt"
    api = "---"
    if summary.is_file():
        for line in summary.read_text(errors="replace").splitlines():
            if "apiVersion" in line and "=" in line:
                api = line.split("=", 1)[1].strip().split()[0]
                break
    if not full.is_file():
        return api, "---"
    supported = "VK_KHR_unified_image_layouts" in full.read_text(errors="replace")
    return api, "yes" if supported else "no"


def build_platform_table(collections: list[Collection]) -> str:
    body = []
    for collection in newest_per_device(collections):
        label = collection.label
        manifest = collection.manifest
        run = manifest["runs"][0]["metadata"]
        driver = run.get("driver_info") or "---"
        driver_name = DRIVER_SHORT.get(
            run.get("driver_name", ""), run.get("driver_name", "")
        )
        driver = f"{driver_name} {driver}".strip().replace("Mesa ", "")
        platform = manifest["platform"].split("-with-")[0]
        # Keep only the kernel/OS version, which is what varies between machines.
        platform = platform.replace("-x86_64", "").replace("-arm64-arm-64bit-Mach-O", "")
        device = DEVICE_SHORT.get(collection.device_name, collection.device_name)
        if collection.mismatched_devices:
            device = " / ".join(
                DEVICE_SHORT.get(name, name)
                for name in sorted(set(collection.devices.values()))
            )
        api, unified = vulkan_facts(collection.root)
        body.append(
            (
                f"\\texttt{{{label}}}",
                device,
                manifest["backend"],
                driver.replace("_", "\\_"),
                api,
                unified,
                platform.replace("_", "\\_"),
            )
        )
    return latex_table(
        caption=(
            "Machines in the fixed 16-pass matrix. Every row is one collection under "
            "\\texttt{paper/data/raw/}; results are never pooled across rows. "
            "``Unified'' reports whether the driver advertises "
            "\\texttt{VK\\_KHR\\_unified\\_image\\_layouts}, which Blade enables when "
            "present."
        ),
        label="tab:platforms",
        column_spec="lllllll",
        header=(
            "Machine",
            "Device",
            "Backend",
            "Driver",
            "API",
            "Unified",
            "Platform",
        ),
        body=body,
        star=True,
    )


def matrix_rows(
    collections: dict[str, Collection],
    metric: str,
    bootstrap_samples: int,
    contenders: Sequence[tuple[str, str]],
    *,
    vulkan_only: bool = True,
) -> list[list[str]]:
    """One block of rows per machine, separated by `\\addlinespace`."""
    blocks: list[list[list[str]]] = []
    for collection in newest_per_device(collections):
        directory = collection.root.name
        if vulkan_only and collection.manifest["backend"] != "vulkan":
            continue
        device = DEVICE_SHORT.get(collection.devices.get("blade", ""), "?")
        block: list[list[str]] = []
        for workload in WORKLOAD_ORDER:
            baseline = collection.values("blade", "automatic", workload, 16, metric)
            if not baseline:
                continue
            cells = [
                device if not block else "",
                WORKLOAD_SHORT[workload],
                f"{median_us(baseline):.1f}",
            ]
            for implementation, policy in contenders:
                # A cross-implementation cell is only meaningful when both
                # implementations ran on the same physical device.
                if implementation != "blade" and collection.mismatched_devices:
                    cells.extend(["n/a", ""])
                    continue
                contender = collection.values(
                    implementation, policy, workload, 16, metric
                )
                if not contender:
                    cells.extend(["---", ""])
                    continue
                difference, low, high = relative_difference(
                    baseline,
                    contender,
                    bootstrap_samples,
                    (directory, workload, metric, implementation, policy),
                )
                cells.append(signed(difference))
                cells.append(interval(low, high))
            block.append(cells)
        if block:
            blocks.append(block)

    rows: list[list[str]] = []
    for index, block in enumerate(blocks):
        if index:
            rows.append([RULE])
        rows.extend(block)
    return rows


def build_gpu_matrix_table(
    collections: dict[str, Collection], bootstrap_samples: int
) -> str:
    contenders = (
        ("blade", "hazard-only"),
        ("blade", "explicit-all"),
        ("wgpu", "tracked"),
    )
    rows = matrix_rows(collections, "gpu_ns", bootstrap_samples, contenders)
    return latex_table(
        caption=(
            "GPU span of one 16-pass command buffer on Vulkan: \\texttt{B-auto} median "
            "in microseconds, then percent differences from it with 95\\% bootstrap "
            "intervals. \\texttt{B-explicit-all} is the instrumentation control and "
            "should be indistinguishable from \\texttt{B-auto}."
        ),
        label="tab:gpu-matrix",
        column_spec="llrrlrlrl",
        header=(
            "Device",
            "Workload",
            "B-auto",
            "\\multicolumn{2}{c}{B-hazard \\%}",
            "\\multicolumn{2}{c}{B-explicit \\%}",
            "\\multicolumn{2}{c}{W-wgpu \\%}",
        ),
        body=rows,
        star=True,
        note=(
            "\\texttt{rubik} selected different devices for the two implementations, "
            "so its \\texttt{W-wgpu} cells are withheld (Section~\\ref{sec:deviations})."
        ),
    )


def build_host_matrix_table(
    collections: dict[str, Collection], bootstrap_samples: int
) -> str:
    contenders = (("blade", "hazard-only"), ("wgpu", "tracked"))
    rows = matrix_rows(
        collections, "host_ns", bootstrap_samples, contenders, vulkan_only=False
    )
    return latex_table(
        caption=(
            "Host cost of one 16-pass command buffer: recording plus submission, "
            "median microseconds, then percent differences from \\texttt{B-auto} with "
            "95\\% bootstrap intervals. These collections have GPU timestamps enabled, "
            "which costs Blade one \\texttt{vkCmdWriteTimestamp} per pass against two "
            "in total for wgpu, so the comparison is conservative for Blade."
        ),
        label="tab:host-matrix",
        column_spec="llrrlrl",
        header=(
            "Device",
            "Workload",
            "B-auto",
            "\\multicolumn{2}{c}{B-hazard \\%}",
            "\\multicolumn{2}{c}{W-wgpu \\%}",
        ),
        body=rows,
        star=True,
        note=(
            "\\texttt{rubik} ran the two implementations on different devices; its host "
            "costs are still comparable in kind but not paired, so the "
            "\\texttt{W-wgpu} column is withheld there as well."
        ),
    )


def build_metal_table(collections: list[Collection]) -> str | None:
    collection = next(
        (c for c in collections if c.manifest["backend"] == "metal"), None
    )
    if collection is None:
        return None
    body = []
    for workload in WORKLOAD_ORDER:
        row = [WORKLOAD_SHORT[workload]]
        for implementation, policy in (("blade", "automatic"), ("wgpu", "tracked")):
            for metric in ("host_ns", "wait_ns"):
                values = collection.values(
                    implementation, policy, workload, 16, metric
                )
                row.append(f"{median_us(values):.1f}" if values else "---")
        body.append(row)
    return latex_table(
        caption=(
            "Apple M3 (Metal), 16 passes: host cost (recording plus submission) and "
            "host wall time waiting for completion, median microseconds. Metal GPU "
            "timestamps are reported per pass rather than as a span and were "
            "unavailable for three of the four wgpu workloads, so wall time is used "
            "instead of device time."
        ),
        label="tab:metal",
        column_spec="lrrrr",
        header=(
            "Workload",
            "Blade host",
            "Blade wait",
            "wgpu host",
            "wgpu wait",
        ),
        body=body,
    )


def build_sweep_table(collections: list[Collection]) -> str | None:
    gpu = next((c for c in collections if c.role == "sweep-gpu"), None)
    cpu = next((c for c in collections if c.role == "sweep-cpu"), None)
    if gpu is None or cpu is None:
        return None
    passes = [count for count in gpu.pass_counts() if count <= 64]
    configurations = (
        ("blade", "automatic", "B-auto"),
        ("blade", "hazard-only", "B-hazard"),
        ("wgpu", "tracked", "W-wgpu"),
    )
    body: list[list[str]] = []
    for workload in ("compute-independent", "graphics-independent"):
        body.append(
            [f"\\multicolumn{{{len(passes) + 3}}}{{l}}{{\\emph{{{workload}}}}}"]
        )
        for source, source_label, metric in (
            (cpu, "host", "host_ns"),
            (gpu, "GPU", "gpu_ns"),
        ):
            for implementation, policy, name in configurations:
                cells = [source_label, name]
                marginal = None
                for count in passes:
                    values = source.values(implementation, policy, workload, count, metric)
                    cells.append(f"{median_us(values):.1f}" if values else "---")
                first = source.values(implementation, policy, workload, passes[0], metric)
                last = source.values(implementation, policy, workload, passes[-1], metric)
                if first and last:
                    marginal = (median_us(last) - median_us(first)) / (
                        passes[-1] - passes[0]
                    )
                cells.append(f"{marginal:.2f}" if marginal is not None else "---")
                body.append(cells)
    return latex_table(
        caption=(
            "Pass-count sweep on \\texttt{zork} (RTX 5070), medians in microseconds. "
            "Host rows come from the timestamp-free collection; GPU rows come from the "
            "GPU-timed collection. The last column is the marginal cost per additional "
            "pass over the measured range."
        ),
        label="tab:sweep",
        column_spec="ll" + "r" * len(passes) + "r",
        header=("", "Config", *(str(count) for count in passes), "$\\mu$s/pass"),
        body=body,
        star=True,
        note=(
            "Both cost components are linear in the pass count over this range, "
            "which is what makes a single marginal cost meaningful."
        ),
    )


def control_floor(collection: Collection, bootstrap_samples: int) -> float:
    """Largest disagreement between `explicit-all` and `automatic` at global
    scope, over the workloads in this collection.

    Those two produce identical command streams, so this is what the device and
    the collection can resolve. Nothing smaller should be claimed on it.
    """
    worst = 0.0
    for workload in WORKLOAD_ORDER:
        baseline = collection.values("blade", "automatic", workload, 16, "gpu_ns")
        control = collection.values("blade", "explicit-all", workload, 16, "gpu_ns")
        if not baseline or not control:
            continue
        difference, _, _ = relative_difference(
            baseline,
            control,
            bootstrap_samples,
            (collection.root.name, workload, "control"),
        )
        worst = max(worst, abs(difference))
    return worst


def build_scope_table(
    collections: list[Collection], bootstrap_samples: int
) -> str | None:
    """Stage/access scope at fixed placement, plus placement for reference."""
    scoped = [c for c in newest_per_device(collections) if c.has_scope_axis]
    if not scoped:
        return None
    rows: list[list[str]] = []
    controls: list[float] = []
    dirty = False
    for index, collection in enumerate(scoped):
        dirty = dirty or collection.is_dirty
        device = DEVICE_SHORT.get(
            collection.devices.get("blade", ""), collection.devices.get("blade", "?")
        )
        floor = control_floor(collection, bootstrap_samples)
        if index:
            rows.append([RULE])
        first = True
        for workload in WORKLOAD_ORDER:
            baseline = collection.values("blade", "automatic", workload, 16, "gpu_ns")
            if not baseline:
                continue
            cells = [
                device if first else "",
                WORKLOAD_SHORT[workload],
                f"{median_us(baseline):.1f}",
                f"{floor:.1f}" if first else "",
            ]
            first = False
            for policy in ("automatic-scoped", "hazard-only", "hazard-only-scoped"):
                contender = collection.values("blade", policy, workload, 16, "gpu_ns")
                if not contender:
                    cells.extend(["---", ""])
                    continue
                difference, low, high = relative_difference(
                    baseline,
                    contender,
                    bootstrap_samples,
                    (collection.root.name, workload, "gpu_ns", policy),
                )
                cells.append(signed(difference))
                cells.append(interval(low, high))
            rows.append(cells)
    note = (
        "The two axes are crossed, so neither is a default for the other. "
        "Columns differ from \\texttt{B-auto} by scope only, by placement only, "
        "and by both. On the chain workloads placement has nothing to remove, "
        "which makes them a scope-only test. The ``both'' column narrows only "
        "the source scope, because an explicitly placed barrier is emitted "
        "where it is written and cannot name a consumer that has not been "
        "declared yet."
    )
    note += (
        " ``ctrl'' is the largest disagreement between \\texttt{explicit-all} "
        "and \\texttt{automatic} at global scope on that device. Those two "
        "emit identical commands, so it is the smallest effect the device and "
        "collection can resolve; a difference below it means nothing."
    )
    if dirty:
        note += (
            " These runs were collected from a modified worktree and are a "
            "pilot, not final data."
        )
    return latex_table(
        caption=(
            "Barrier placement crossed with barrier scope, 16 passes: "
            "\\texttt{B-auto} GPU-span median in microseconds, then percent "
            "differences from it with 95\\% bootstrap intervals."
        ),
        label="tab:scope",
        column_spec="llrrrlrlrl",
        header=(
            "Device",
            "Workload",
            "B-auto",
            "ctrl",
            "\\multicolumn{2}{c}{scope \\%}",
            "\\multicolumn{2}{c}{placement \\%}",
            "\\multicolumn{2}{c}{both \\%}",
        ),
        body=rows,
        star=True,
        note=note,
    )


def build_summary_csv(collections: list[Collection], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            (
                "collection",
                "machine",
                "role",
                "device",
                "implementation",
                "policy",
                "workload",
                "passes",
                "samples",
                "host_us_median",
                "gpu_us_median",
                "wait_us_median",
            )
        )
        for collection in collections:
            directory, label, role = (
                collection.root.name,
                collection.label,
                collection.role,
            )
            for key in sorted(collection.samples, key=str):
                implementation, policy, workload, passes = key
                bucket = collection.samples[key]
                writer.writerow(
                    (
                        directory,
                        label,
                        role,
                        collection.devices.get(implementation, ""),
                        implementation,
                        policy,
                        workload,
                        passes,
                        len(bucket["gpu_ns"]),
                        f"{median_us(bucket['host_ns']):.3f}",
                        f"{median_us(bucket['gpu_ns']):.3f}",
                        f"{median_us(bucket['wait_ns']):.3f}",
                    )
                )


def main() -> None:
    arguments = parse_arguments()
    arguments.output.mkdir(parents=True, exist_ok=True)
    collections = discover_collections(arguments.raw)
    if not collections:
        raise ValueError(f"no collections found under {arguments.raw}")

    for collection in collections:
        print(f"{collection.root.name}: {collection.label} ({collection.role})")
        if collection.mismatched_devices:
            print(
                f"  warning: implementations used different devices: "
                f"{collection.devices}"
            )
        if collection.is_dirty:
            print("  warning: collected from a modified worktree")

    outputs = {
        "platforms.tex": build_platform_table(collections),
        "gpu-matrix.tex": build_gpu_matrix_table(
            collections, arguments.bootstrap_samples
        ),
        "host-matrix.tex": build_host_matrix_table(
            collections, arguments.bootstrap_samples
        ),
        "metal.tex": build_metal_table(collections),
        "sweep.tex": build_sweep_table(collections),
        "scope.tex": build_scope_table(collections, arguments.bootstrap_samples),
    }
    written = 0
    for name, content in outputs.items():
        if content is None:
            print(f"skipping {name}: no collection supplies it")
            continue
        (arguments.output / name).write_text(content, encoding="utf-8")
        written += 1
    build_summary_csv(collections, arguments.output / "summary.csv")
    print(f"Wrote {written + 1} files to {arguments.output}")


if __name__ == "__main__":
    main()

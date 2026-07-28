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
import hashlib
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

# The paper names systems, not the machines they happened to run on. Hostnames
# are an accident of whose desk a box sits on and carry no information a reader
# can use; `S1`-`S5` follow `MATRIX_ORDER`, so the numbering is the order the
# results are presented in.
MACHINE_LABEL = {host: f"S{index + 1}" for index, host in enumerate(MATRIX_ORDER)}


def machine_label(host: str) -> str:
    """Reader-facing name for a collection's machine."""
    short = host.split(".", 1)[0]
    return MACHINE_LABEL.get(short.lower(), short)


DEVICE_SHORT = {
    "NVIDIA GeForce RTX 5070": "RTX 5070",
    "AMD Radeon RX 7900 XT (RADV NAVI31)": "RX 7900 XT",
    "AMD Ryzen 5 9600X 6-Core Processor (RADV RAPHAEL_MENDOCINO)": "Raphael iGPU",
    "AMD Radeon 780M Graphics (RADV PHOENIX)": "Radeon 780M",
    "Intel(R) Graphics (RPL-U)": "Intel Xe (RPL-U)",
    "Apple M3": "Apple M3",
}

# Keys the prose uses to name a device. The paper cites roughly a hundred
# numbers; typing them by hand is how a table gets regenerated and the
# sentence beside it does not.
DEVICE_SLUG = {
    "NVIDIA GeForce RTX 5070": "rtx5070",
    "AMD Radeon RX 7900 XT (RADV NAVI31)": "rx7900xt",
    "AMD Ryzen 5 9600X 6-Core Processor (RADV RAPHAEL_MENDOCINO)": "raphael",
    "AMD Radeon 780M Graphics (RADV PHOENIX)": "radeon780m",
    "Intel(R) Graphics (RPL-U)": "intelxe",
    "Apple M3": "applem3",
}

# (macro suffix, implementation, policy) for every comparison against B-auto.
COMPARISONS = (
    ("placement", "blade", "hazard-only"),
    ("scope", "blade", "automatic-scoped"),
    ("both", "blade", "hazard-only-scoped"),
    ("control", "blade", "explicit-all"),
    ("wgpu", "wgpu", "tracked"),
)

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

# A machine's host measurements are used when the median 16-pass block spreads
# no more than this, relative to its own median. Every machine in this study is
# either well under it or three times over it, so the value is a separator
# rather than a tuning knob.
HOST_DISPERSION_LIMIT = 40.0

# Host medians on the sweep driver turn unstable past this pass count --- the
# 32- and 64-pass host columns are non-monotonic --- so host endpoint averages
# use only the lower, monotone region. GPU endpoint averages use the whole
# range, whose increments are nearly constant.
HOST_MARGINAL_CAP = 8

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
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "generate partial audit outputs even when main.tex cites a device "
            "whose raw collection is absent; such outputs cannot build the paper"
        ),
    )
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
        manifest = root / "manifest.json"
        if not manifest.is_file():
            continue
        document = json.loads(manifest.read_text())
        schema = document.get("schema", "")
        if schema != "blade-sync-study-v1":
            # Profiles and captures live alongside the timing collections but
            # are read by their own table builders.
            continue
        if document.get("parameters", {}).get("validation", False):
            # Correctness collections deliberately use tiny shapes, one
            # repetition, and validation layers. They are evidence for the
            # implementation, never performance input.
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


def matrix_rank(collection: Collection) -> int:
    """How well a collection can stand in for the fixed 16-pass matrix.

    A dedicated matrix collection is the purpose-built source. Failing that, a
    GPU-timed sweep that includes the 16-pass point serves: its 16-pass blocks
    are the same protocol --- one randomized process launch per configuration,
    the same warm-up and sample counts --- collected alongside the other
    counts. A timestamp-free sweep cannot supply device time, so it never
    stands in.
    """
    if collection.role == "matrix":
        return 2
    if collection.role == "sweep-gpu" and 16 in collection.pass_counts():
        return 1
    return 0


def newest_per_device(collections: list[Collection]) -> list[Collection]:
    """One fixed-matrix source per machine and device.

    A retest supersedes an earlier run of the same hardware rather than
    appearing beside it, so the tables never show one device twice. A
    dedicated matrix collection wins over a sweep standing in for one; within
    a rank, the most recent wins.
    """
    best: dict[tuple[str, str], Collection] = {}
    for collection in collections:
        rank = matrix_rank(collection)
        if not rank:
            continue
        key = (collection.label, collection.device_name)
        previous = best.get(key)
        if previous is None or (rank, collection.collected_utc) > (
            matrix_rank(previous),
            previous.collected_utc,
        ):
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
        # The thirty samples in one CSV share a process launch, clocks and
        # thermal state. Preserve that process-level block so comparisons do
        # not pretend that all ninety observations are independent.
        self.block_samples: dict[
            tuple, dict[str, dict[str, list[float]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.devices: dict[str, str] = {}
        # Whether the wgpu side ran with its injected bounds, division and loop
        # checks. Those cost tens of percent of GPU span on a fragment-bound
        # workload, so a collection taken with them is not comparable with one
        # taken without, and the two must never share a column. Collections
        # predating the flag do not record it and were all taken with checks on.
        self.wgpu_shader_checks = True
        # Per-block clock drift, keyed by metric. One block is one process
        # launch, so the device idles between blocks and ramps again inside
        # each; pooling the samples hides that, which is why it is measured
        # here while the file is still in order.
        self.drift: dict[str, list[float]] = defaultdict(list)
        for path in sorted(root.glob("r*.csv")):
            run = analyze.read_run(path)
            implementation = run.metadata["implementation"]
            repetition = run.metadata.get("repetition", path.name)
            self.devices[implementation] = run.metadata["device_name"]
            if implementation == "wgpu":
                self.wgpu_shader_checks = (
                    run.metadata.get("shader_checks", "true").lower() != "false"
                )
            for metric in ("gpu_ns", "wait_ns"):
                ordered = [
                    float(row[metric]) for row in run.rows if int(row["passes"]) == 16
                ]
                third = len(ordered) // 3
                if third < 3:
                    continue
                first = statistics.median(ordered[:third])
                last = statistics.median(ordered[-third:])
                if first:
                    self.drift[metric].append(100.0 * (last - first) / first)
            for row in run.rows:
                key = (
                    implementation,
                    row["policy"],
                    row["workload"],
                    int(row["passes"]),
                )
                bucket = self.samples[key]
                block = self.block_samples[key][repetition]
                for metric in analyze.METRICS:
                    value = float(row[metric])
                    bucket[metric].append(value)
                    block[metric].append(value)
                host = float(row["record_ns"]) + float(row["submit_ns"])
                bucket["host_ns"].append(host)
                block["host_ns"].append(host)
        if self.role is None:
            self.role = self._infer_role()

    def _infer_label(self) -> str:
        host = self.manifest.get("host")
        if host:
            return str(host).split(".")[0].lower()
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
    def display_label(self) -> str:
        return machine_label(self.label)

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

    def blocked_values(
        self, implementation: str, policy: str, workload: str, passes: int, metric: str
    ) -> dict[str, list[float]]:
        """Samples by process-level repetition for one configuration."""
        blocks = self.block_samples.get(
            (implementation, policy, workload, passes), {}
        )
        return {
            repetition: metrics[metric]
            for repetition, metrics in blocks.items()
            if metrics.get(metric)
        }

    def pass_counts(self) -> list[int]:
        return sorted({key[3] for key in self.samples})

    def dispersion(self, metric: str) -> float | None:
        """Median interquartile range of a 16-pass block, relative to its median.

        The control floor catches noise that both control configurations saw,
        because they were measured against each other. It cannot catch noise
        confined to one block: each (policy, workload) pair is measured as a
        run of samples, so interference lasting a few seconds lands entirely
        inside one configuration and produces a tight interval around a wrong
        value. Within-block dispersion is what sees that, and it is what
        disqualifies one machine's host measurements here.
        """
        spreads = []
        for key, blocks in self.block_samples.items():
            if key[3] != 16:
                continue
            for block in blocks.values():
                values = sorted(block.get(metric, []))
                if len(values) < 4:
                    continue
                quartiles = statistics.quantiles(values, n=4)
                median = statistics.median(values)
                if median:
                    spreads.append(100.0 * (quartiles[2] - quartiles[0]) / median)
        return statistics.median(spreads) if spreads else None


def process_median_us(
    collection: Collection,
    implementation: str,
    policy: str,
    workload: str,
    passes: int,
    metric: str,
) -> float | None:
    """Median of the separately launched process medians, in microseconds."""
    blocks = collection.blocked_values(
        implementation, policy, workload, passes, metric
    )
    if not blocks:
        return None
    return (
        statistics.median(statistics.median(values) for values in blocks.values())
        / 1000.0
    )


_INTERVALS: dict[tuple, tuple[float, float, float]] = {}


def comparison(
    collection: Collection,
    workload: str,
    metric: str,
    implementation: str,
    policy: str,
    bootstrap_samples: int,
    *,
    against: tuple[str, str] = ("blade", "automatic"),
    passes: int = 16,
) -> tuple[float, float, float] | None:
    """Paired percent difference from `against`, with a hierarchical interval.

    The seed is derived here rather than at each call site, so every table and
    every number in the prose that names the same comparison gets the same
    resampling and therefore the same interval. Two builders passing seeds of
    different shapes for one quantity is how a table and the sentence beside it
    come to disagree in the last digit.
    """
    key = (
        collection.root.name,
        workload,
        metric,
        implementation,
        policy,
        against,
        passes,
        bootstrap_samples,
    )
    if key in _INTERVALS:
        return _INTERVALS[key]
    baseline = collection.blocked_values(*against, workload, passes, metric)
    contender = collection.blocked_values(
        implementation, policy, workload, passes, metric
    )
    if not baseline or not contender:
        return None
    _, result = analyze.paired_hierarchical_intervals(
        baseline,
        contender,
        bootstrap_samples,
        tuple(str(part) for part in key[:-1]),
    )
    _INTERVALS[key] = result
    return result


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
    preamble_rows: Sequence[str] = (),
    column_sep: str | None = None,
) -> str:
    environment = "table*" if star else "table"
    lines = [
        f"% Generated by paper/build-tables.py -- do not edit by hand.",
        f"\\begin{{{environment}}}[t]",
        "\\centering",
        "\\small",
        # A sixteen-column table only fits the page with narrower inter-column
        # space; the group ends with the environment, so nothing leaks.
        *([f"\\setlength{{\\tabcolsep}}{{{column_sep}}}"] if column_sep else []),
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        # Verbatim lines above the column names, for a spanning header.
        *preamble_rows,
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
        label = collection.display_label
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
        # The kernel or Darwin version is the part that differs between
        # machines; the rest of `platform.platform()` is the same words again.
        platform = platform.split("-with-")[0]
        for prefix in ("Linux-", "macOS-"):
            if platform.startswith(prefix):
                platform = prefix.rstrip("-") + " " + platform[len(prefix) :]
        platform = platform.split("-generic")[0].split("-arm64")[0]
        body.append(
            (
                f"\\texttt{{{label}}}",
                device,
                manifest["backend"],
                driver.replace("_", "\\_"),
                api,
                unified,
                platform.replace("_", "\\_"),
                f"\\texttt{{{collection.blade_revision}}}",
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
        column_spec="@{}llllllll@{}",
        header=(
            "Machine",
            "Device",
            "Backend",
            "Driver",
            "API",
            "Unified",
            "Platform",
            "Revision",
        ),
        body=body,
        star=True,
        note=(
            "The Vulkan rows were all collected on one day from one benchmark "
            "build: where their revisions differ, they differ only by commits "
            "under \\texttt{paper/}, which the benchmark does not compile, and "
            "every row used the same \\wgpu{} revision. The Metal harness has "
            "only its backend's automatic policy; the scope variants in this "
            "study alter Vulkan barriers and therefore do not apply to that row."
        ),
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
            baseline = process_median_us(
                collection, "blade", "automatic", workload, 16, metric
            )
            if baseline is None:
                continue
            cells = [
                device if not block else "",
                WORKLOAD_SHORT[workload],
                f"{baseline:.1f}",
            ]
            for implementation, policy in contenders:
                # A cross-implementation cell is only meaningful when both
                # implementations ran on the same physical device and, for the
                # device-side metrics, compiled their shaders the same way.
                # The injected checks change what the GPU executes, not what
                # the host records, so host cells survive a stale wgpu.
                if implementation != "blade" and (
                    collection.mismatched_devices
                    or (collection.wgpu_shader_checks and metric != "host_ns")
                ):
                    cells.extend(["n/a", ""])
                    continue
                result = comparison(
                    collection,
                    workload,
                    metric,
                    implementation,
                    policy,
                    bootstrap_samples,
                )
                if result is None:
                    cells.extend(["---", ""])
                    continue
                difference, low, high = result
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
    # Only say a column is withheld if one actually is. The first `rubik`
    # collection had the two implementations on different devices; the
    # recollection does not, and a note that outlives its cause is a claim
    # about data that is no longer there.
    # Only machines that would otherwise have cells in this Vulkan-only table
    # belong in its withholding note; the Metal row is not in it to withhold.
    vulkan = [
        c
        for c in newest_per_device(collections)
        if c.manifest["backend"] == "vulkan"
    ]
    withheld = [c for c in vulkan if c.mismatched_devices]
    stale = [c for c in vulkan if c.wgpu_shader_checks]
    return latex_table(
        caption=(
            "GPU span of one 16-pass command buffer on Vulkan: \\texttt{B-auto} "
            "median of process medians in microseconds, then the median paired "
            "process-level percent differences with 95\\% "
            "hierarchical-bootstrap intervals. "
            "\\texttt{B-explicit-all} is the instrumentation control: it "
            "issues the same timed global barrier calls as \\texttt{B-auto} by "
            "construction. The far endpoint of its interval is used as a "
            "conservative, post-hoc stability threshold "
            "(Section~\\ref{sec:deviations}). \\texttt{W-wgpu} is an "
            "end-to-end comparison with two timestamps ending at the final "
            "pass, whereas \\blade{} writes one per pass and includes its final "
            "barrier in the span."
        ),
        label="tab:gpu-matrix",
        column_spec="@{}llr" + "r@{\\,}l" * 3 + "@{}",
        header=(
            "Device",
            "Workload",
            "B-auto",
            "\\multicolumn{2}{c}{B-hazard \\%}",
            "\\multicolumn{2}{c}{B-exp-all \\%}",
            "\\multicolumn{2}{c}{W-wgpu \\%}",
        ),
        body=rows,
        star=True,
        note=(
            (
                ", ".join(f"\\texttt{{{c.display_label}}}" for c in withheld)
                + " selected different devices for the two implementations, so "
                "its \\texttt{W-wgpu} cells are withheld "
                "(Section~\\ref{sec:deviations})."
            )
            if withheld
            else (
                "Every cell compares the two implementations on the same "
                "physical device; the collector aborts otherwise "
                "(Section~\\ref{sec:deviations})."
            )
        )
        + (
            (
                " \\texttt{W-wgpu} is withheld on "
                + ", ".join(f"\\texttt{{{c.display_label}}}" for c in stale)
                + ", whose \\wgpu{} side still carried the injected shader "
                "checks of Section~\\ref{sec:workloads}; those cells measure "
                "code generation as much as synchronization and are not "
                "comparable with the rest."
            )
            if stale
            else ""
        ),
    )


def build_host_matrix_table(
    collections: dict[str, Collection], bootstrap_samples: int
) -> str:
    contenders = (("blade", "hazard-only"), ("wgpu", "tracked"))
    rows = matrix_rows(
        collections, "host_ns", bootstrap_samples, contenders, vulkan_only=False
    )
    spreads = {
        collection.display_label: spread
        for collection in newest_per_device(collections)
        if (spread := collection.dispersion("host_ns")) is not None
    }
    unstable = {
        label: spread
        for label, spread in spreads.items()
        if spread > HOST_DISPERSION_LIMIT
    }
    stable = [spread for label, spread in spreads.items() if label not in unstable]
    dispersion_note = ""
    if unstable and stable:
        dispersion_note = (
            " Host cost on "
            + ", ".join(f"\\texttt{{{label}}}" for label in sorted(unstable))
            + " is not measured by this collection and is shown only for "
            "completeness: at the median 16-pass block its samples spread "
            + ", ".join(f"{spread:.0f}\\%" for spread in sorted(unstable.values()))
            + " of their own median, against at most "
            f"{max(stable):.0f}\\% on every other machine, while its "
            "\\emph{device} blocks are unaffected. That is CPU-side "
            "interference confined to individual blocks, which a control "
            "floor cannot see because both controls sit inside it."
        )
    return latex_table(
        caption=(
            "Host cost of one 16-pass command buffer: recording plus submission, "
            "median of process medians in microseconds, then percent differences "
            "from \\texttt{B-auto} with 95\\% hierarchical-bootstrap intervals "
            "over paired process repetitions. These collections have GPU "
            "timestamps enabled. Blade "
            "records one \\texttt{vkCmdWriteTimestamp} per pass against two in "
            "total for wgpu, so this is an instrumented end-to-end comparison, "
            "not a matched timestamp workload."
        ),
        label="tab:host-matrix",
        column_spec="@{}llr" + "r@{\\,}l" * 2 + "@{}",
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
            "Percent differences rather than ratios: a \\texttt{W-wgpu} entry of "
            "$+200\\%$ is three times \\texttt{B-auto}." + dispersion_note
        ),
    )


def build_metal_table(collections: list[Collection]) -> str | None:
    collection = max(
        (c for c in collections if c.manifest["backend"] == "metal"),
        key=lambda c: c.collected_utc,
        default=None,
    )
    if collection is None:
        return None
    body = []
    for workload in WORKLOAD_ORDER:
        row = [WORKLOAD_SHORT[workload]]
        measured = False
        for implementation, policy in (("blade", "automatic"), ("wgpu", "tracked")):
            for metric in ("host_ns", "wait_ns"):
                value = process_median_us(
                    collection, implementation, policy, workload, 16, metric
                )
                row.append(f"{value:.1f}" if value is not None else "---")
                measured = measured or value is not None
        # Older Metal collections predate mixed workloads, so their rows would
        # otherwise be four dashes claiming a measurement was attempted.
        if measured:
            body.append(row)
    note = None
    if collection.wgpu_shader_checks:
        note = (
            "This collection predates the shader-parity fix of "
            "Section~\\ref{sec:workloads}: the \\wgpu{} waits include its "
            "injected shader checks, so the wait comparison overstates "
            "\\blade's advantage. Host times are recording work and are "
            "unaffected."
        )
    return latex_table(
        caption=(
            "Apple M3 (Metal), 16 passes: host cost (recording plus submission) and "
            "host wall time waiting for completion, median of process medians in "
            "microseconds. Metal GPU timestamps are reported per pass rather than "
            "as a span and were unavailable for three of the four wgpu workloads, "
            "so wall time is used instead of device time."
        ),
        label="tab:metal",
        column_spec="@{}lrrrr@{}",
        header=("Workload", "host", "wait", "host", "wait"),
        preamble_rows=(
            " & \\multicolumn{2}{c}{\\blade{}} & \\multicolumn{2}{c}{\\wgpu{}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        ),
        body=body,
        note=note,
    )


def newest_sweeps(collections: list[Collection], role: str) -> list[Collection]:
    """The newest sweep of `role` for each machine and device, in table order."""
    best: dict[tuple[str, str], Collection] = {}
    for collection in collections:
        if collection.role != role:
            continue
        key = (collection.label, collection.device_name)
        if key not in best or collection.collected_utc > best[key].collected_utc:
            best[key] = collection
    def rank(key: tuple[str, str]) -> tuple:
        try:
            order = MATRIX_ORDER.index(key[0])
        except ValueError:
            order = len(MATRIX_ORDER)
        return (order, key)
    return [best[key] for key in sorted(best, key=rank)]


def sweep_pair(collections: list[Collection]) -> tuple[Collection, Collection] | None:
    """The newest GPU-timed and timestamp-free sweeps of one device.

    The scaling table reads host cost against device cost, so the two sides
    must come from the same machine and device; a GPU sweep from one machine
    must never be paired with a host sweep from another. When several devices
    have both sweeps, the earliest in `MATRIX_ORDER` is the one the scaling
    section describes.
    """
    gpu = {
        (c.label, c.device_name): c for c in newest_sweeps(collections, "sweep-gpu")
    }
    cpu = {
        (c.label, c.device_name): c for c in newest_sweeps(collections, "sweep-cpu")
    }
    shared = [key for key in gpu if key in cpu]
    if not shared:
        return None
    def rank(key: tuple[str, str]) -> tuple:
        try:
            order = MATRIX_ORDER.index(key[0])
        except ValueError:
            order = len(MATRIX_ORDER)
        return (order, key)
    key = min(shared, key=rank)
    return gpu[key], cpu[key]


def build_sweep_table(collections: list[Collection]) -> str | None:
    pair = sweep_pair(collections)
    if pair is None:
        return None
    gpu, cpu = pair
    device = DEVICE_SHORT.get(
        gpu.devices.get("blade", ""), gpu.devices.get("blade", "?")
    ).replace(" ", "~")
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
        for source, source_label, metric, marginal_cap in (
            (cpu, "host", "host_ns", HOST_MARGINAL_CAP),
            (gpu, "GPU", "gpu_ns", 64),
        ):
            for implementation, policy, name in configurations:
                cells = [source_label, name]
                for count in passes:
                    value = process_median_us(
                        source, implementation, policy, workload, count, metric
                    )
                    cells.append(f"{value:.1f}" if value is not None else "---")
                marginal = marginal_cost(
                    source, implementation, policy, workload, metric, marginal_cap
                )
                cells.append(f"{marginal:.2f}" if marginal is not None else "---")
                body.append(cells)
    return latex_table(
        caption=(
            f"Pass-count sweep on the {device}, medians of process medians in "
            "microseconds. "
            "Host rows come from the timestamp-free collection; GPU rows come from the "
            "GPU-timed collection. The last column is the average increment per "
            "additional pass between the endpoint medians, taken over the whole "
            "range for the GPU rows and over 1--"
            f"{HOST_MARGINAL_CAP} passes for the host rows."
        ),
        label="tab:sweep",
        column_spec="@{}ll" + "r" * len(passes) + "r@{}",
        header=("", "Config", *(str(count) for count in passes), "$\\mu$s/pass"),
        body=body,
        star=True,
        note=(
            "GPU span has nearly constant increments across the whole range. "
            "Host cost is linear only to "
            f"{HOST_MARGINAL_CAP} passes: between "
            f"{HOST_MARGINAL_CAP} and 16 its medians jump several-fold into a "
            "second, steeper regime, so one slope fitted over all counts "
            "would describe neither. The 1--"
            f"{HOST_MARGINAL_CAP} average characterizes the linear regime; "
            "the larger counts are reported as measured."
        ),
    )


def control_floor(
    collection: Collection, workload: str, bootstrap_samples: int
) -> float | None:
    """A conservative post-hoc stability threshold for one cell.

    `explicit-all` and `automatic` issue the same global barrier calls, so
    their device-time disagreement diagnoses measurement instability. The
    threshold is the far end of the control's interval rather than its point
    estimate. This matters when process repetitions land in different clock or
    interference regimes: a small point estimate can then have a wide
    hierarchical interval. This heuristic is deliberately conservative; it is
    not a confidence bound or a preregistered equivalence margin.

    It is computed per workload rather than per device because noise is not a
    property of the device alone; a per-device maximum would throw away stable
    cells merely because another workload on the same machine varied.
    """
    result = comparison(
        collection, workload, "gpu_ns", "blade", "explicit-all", bootstrap_samples
    )
    if result is None:
        return None
    _, low, high = result
    return max(abs(low), abs(high))


def clears_stability_floor(
    result: tuple[float, float, float], floor: float
) -> bool:
    """Whether an effect interval lies wholly beyond its control floor.

    Requiring only a one-signed interval and a point estimate beyond the floor
    is internally inconsistent: such an interval may still substantially
    overlap the range reached by the identical-request control. The stricter
    rule used here requires the near endpoint of the effect interval to clear
    the same-side floor.
    """
    _, low, high = result
    return low > floor or high < -floor




# Display order for the profile table. Every bucket `profile-hosts.py` can
# emit must appear here: a bucket absent from this tuple is dropped silently,
# and the columns then no longer sum to the process. That happened once, to a
# row worth 76%.
PROFILE_BUCKET_ORDER = (
    "kernel",
    "driver",
    "ash / loader",
    "wgpu tracker",
    "wgpu init tracker",
    "wgpu command",
    "wgpu-hal",
    "wgpu device/resource",
    "wgpu validation",
    "wgpu other",
    "blade",
    "allocator",
    "libc / runtime",
    "other",
)


class Profile:
    """One `profile-hosts.py` collection: bucket shares per workload."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.host = machine_label(str(self.manifest.get("host", ""))) or root.name
        self.collected_utc = str(self.manifest.get("created_utc", ""))
        self.shares: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
        for row in csv.DictReader((root / "buckets.csv").open()):
            if row["bucket"].startswith("TOTAL"):
                continue
            self.shares[row["workload"], row["implementation"]][row["bucket"]] = float(
                row["self_percent"]
            )

    @property
    def workloads(self) -> list[str]:
        present = {key[0] for key in self.shares}
        return [w for w in WORKLOAD_ORDER if w in present]

    @property
    def buckets(self) -> set[str]:
        return {bucket for shares in self.shares.values() for bucket in shares}

    @property
    def devices_by_implementation(self) -> dict[str, set[str]]:
        selected: dict[str, set[str]] = defaultdict(set)
        for key, device in self.manifest.get("devices", {}).items():
            implementation, _, _workload = key.partition("/")
            if implementation and device:
                selected[implementation].add(DEVICE_SHORT.get(device, device))
        return selected

    @property
    def driver_library(self) -> str:
        """The shared object that supplied the `driver` bucket.

        The retained profiling invocations did not pin an adapter. The driver
        can still be named because it is the object the samples landed in, but
        the manifest's per-run device report is needed to detect a multi-GPU
        mismatch between implementations.
        """
        weights: dict[str, float] = defaultdict(float)
        for row in csv.DictReader((self.root / "symbols.csv").open()):
            if row["bucket"] == "driver":
                weights[row["dso"]] += float(row["self_percent"])
        return max(weights, key=weights.get, default="")


def newest_profile_per_host(raw: Path) -> list[Profile]:
    """One profile per machine, the most recent, in machine order.

    A re-profiled machine supersedes its earlier run rather than appearing
    beside it as a second column with the same name.
    """
    best: dict[str, Profile] = {}
    for directory in sorted(raw.glob("*-profile")):
        if not (directory / "buckets.csv").is_file():
            continue
        if not (directory / "manifest.json").is_file():
            continue
        profile = Profile(directory)
        previous = best.get(profile.host)
        if previous is None or profile.collected_utc > previous.collected_utc:
            best[profile.host] = profile
    return sorted(best.values(), key=lambda p: p.host)


def build_profile_table(raw: Path) -> str | None:
    """Where each implementation's host CPU time goes, from `profile-hosts.py`.

    Reported as a share of the process, not as absolute time between
    implementations: `task-clock` counts a blocking fence wait as CPU time when
    the driver spins in it, so the two totals are not comparable. The share of
    a single process among its own components is unaffected by that.

    Every profile on disk is reported. Two machines disagree about the tracker
    share by a factor of three, which is a result rather than a detail to pick
    a winner from.
    """
    profiles = newest_profile_per_host(raw)
    if not profiles:
        return None
    unknown = sorted(
        bucket
        for profile in profiles
        for bucket in profile.buckets
        if bucket not in PROFILE_BUCKET_ORDER
    )
    if unknown:
        raise ValueError(
            "profile buckets missing from PROFILE_BUCKET_ORDER, which would "
            f"drop them from the table without saying so: {unknown}"
        )

    columns: list[tuple[Profile, str, str]] = [
        (profile, workload, implementation)
        for profile in profiles
        for workload in profile.workloads
        for implementation in ("blade", "wgpu")
    ]
    body = []
    for bucket in PROFILE_BUCKET_ORDER:
        cells = [bucket.replace("_", "\\_")]
        present = False
        for profile, workload, implementation in columns:
            value = profile.shares.get((workload, implementation), {}).get(bucket)
            if value is None or value < 0.05:
                cells.append("---")
            else:
                cells.append(f"{value:.1f}")
                present = True
        if present:
            body.append(cells)

    # Three header levels --- machine, workload, implementation --- so a
    # column is only as wide as its numbers. One level of `b c-ind` headers
    # across sixteen columns is what made this table wider than the page.
    machine_row, workload_row = [""], [""]
    machine_rules, workload_rules = [], []
    index = 2
    for profile in profiles:
        width = 2 * len(profile.workloads)
        machine_row.append(
            f"\\multicolumn{{{width}}}{{c}}{{\\texttt{{{profile.host}}}}}"
        )
        machine_rules.append(f"\\cmidrule(lr){{{index}-{index + width - 1}}}")
        for workload in profile.workloads:
            workload_row.append(
                f"\\multicolumn{{2}}{{c}}{{{WORKLOAD_SHORT.get(workload, workload)}}}"
            )
            workload_rules.append(f"\\cmidrule(lr){{{index}-{index + 1}}}")
            index += 2
    header = ["Component"] + [
        f"\\textsc{{{implementation[0]}}}"
        for _, _, implementation in columns
    ]
    drivers = ", ".join(
        f"\\texttt{{{profile.host}}} on \\texttt{{{profile.driver_library}}}"
        for profile in profiles
        if profile.driver_library
    ).replace("_", "\\_")
    adapter_mismatches = []
    for profile in profiles:
        devices = profile.devices_by_implementation
        selected = {device for names in devices.values() for device in names}
        if len(selected) <= 1:
            continue
        implementations = ", ".join(
            f"\\textsc{{{implementation[0]}}}=" + "/".join(sorted(names))
            for implementation, names in sorted(devices.items())
        )
        adapter_mismatches.append(
            f"\\texttt{{{profile.host}}} selected {implementations}"
        )
    adapter_note = (
        " The retained profiles were not adapter-pinned; "
        + "; ".join(adapter_mismatches)
        + ", so those side-by-side columns are not a matched-device profile."
        if adapter_mismatches
        else ""
    )
    return latex_table(
        caption=(
            "Share of process CPU time by component, from flat \\texttt{perf} "
            "profiles. \\textsc{b} is \\blade{} and \\textsc{w} the corresponding "
            "\\wgpu{} program. The workload uses tiny dispatches, small targets, "
            "many passes, and no timestamp queries, but the profile covers the "
            "whole process, including completion waits."
        ),
        label="tab:profile",
        column_spec="@{}l" + "r" * len(columns) + "@{}",
        header=header,
        body=body,
        star=True,
        column_sep="3.5pt",
        preamble_rows=(
            " & ".join(machine_row) + " \\\\",
            "".join(machine_rules),
            " & ".join(workload_row) + " \\\\",
            "".join(workload_rules),
        ),
        note=(
            "Shares within a process, not times between processes: "
            "\\texttt{task-clock} charges a blocking fence wait to the process "
            "whenever the driver spins inside it, which inflates the "
            "\\emph{driver} row differently for the two implementations. Self "
            "time is attributed to the symbol a sample landed in, so inlined "
            "tracker work can be charged to its caller; the tracker-labelled "
            "share can therefore undercount, but is not a formal lower bound. "
            "Columns sum to slightly more or less than 100 because "
            "\\texttt{perf} rounds each symbol's share independently. "
            f"The \\emph{{driver}} row is {drivers}.{adapter_note} "
            "The columns are labelled by machine and driver rather than by "
            "device; only shares within one process are read from them. "
            "Because waits dominate these whole-process profiles, the "
            "component shares cannot apportion the record-and-submit gap."
        ),
    )


NUMBERS_PREAMBLE = r"""% Generated by paper/build-tables.py -- do not edit by hand.
%
% Every measured number the prose quotes is defined here and cited by key, so
% a regenerated table can never disagree with the sentence beside it. A key
% with no definition raises an error rather than typesetting as nothing.
\makeatletter
\newcommand{\bladenum}[1]{%
  \ifcsname blade@num@#1\endcsname
    \csname blade@num@#1\endcsname
  \else
    \errmessage{No generated number for `#1'; run paper/build-tables.py}%
  \fi}
\makeatother
% All prefixed `b`: \mag is a TeX primitive and \floor is taken by amsmath in
% some configurations, and a clash here is a build error rather than a wrong
% number only because LaTeX happens to check.
% Signed percent difference from B-auto, e.g. -26.3\%.
\newcommand{\bpct}[3]{\bladenum{#1/#2/#3/pct}\%}
% The same magnitude without its sign, for "worth 26.3\%" phrasing.
\newcommand{\bmag}[3]{\bladenum{#1/#2/#3/mag}\%}
% 95\% paired hierarchical-bootstrap interval, signed and unsigned to match
% the two above.
\newcommand{\bpctci}[3]{\bpct{#1}{#2}{#3}\,[\bladenum{#1/#2/#3/lo},\,\bladenum{#1/#2/#3/hi}]}
\newcommand{\bmagci}[3]{\bmag{#1}{#2}{#3}\,[\bladenum{#1/#2/#3/maglo},\,\bladenum{#1/#2/#3/maghi}]}
% The post-hoc stability threshold of one cell: the largest disagreement
% between two configurations issuing the same timed global barrier calls that
% the control interval reaches.
\newcommand{\bfloor}[2]{\bladenum{#1/#2/floor}\%}
% A median in microseconds: device, workload, one of
% auto/placement/scope/both/control/wgpu, optionally prefixed `host`. Carries
% its own math mode, so it is written in text and not inside dollars.
\newcommand{\bus}[3]{\bladenum{#1/#2/#3us}\,$\mu$s}
"""


def numbers_macros(
    collections: list[Collection], raw: Path, bootstrap_samples: int
) -> str:
    """Define every measured number the prose cites, keyed by name.

    The alternative is copying numbers into sentences by hand, which survives
    exactly until the next collection: at the time this was written the paper
    quoted three placement effects that the tables no longer supported.
    """
    values: dict[str, str] = {}

    def record(key: str, difference: float, low: float, high: float) -> None:
        values[f"{key}/pct"] = f"{difference:+.1f}"
        values[f"{key}/lo"] = f"{low:+.1f}"
        values[f"{key}/hi"] = f"{high:+.1f}"
        # The unsigned form says "worth 26.3% [20.1, 28.0]", which is only a
        # sentence one may write when the interval has a sign. Leaving these
        # keys undefined for a cell whose interval spans zero makes `\magci`
        # fail the build there rather than print a nonsense range.
        if low > 0 or high < 0:
            values[f"{key}/mag"] = f"{abs(difference):.1f}"
            magnitudes = sorted((abs(low), abs(high)))
            values[f"{key}/maglo"] = f"{magnitudes[0]:.1f}"
            values[f"{key}/maghi"] = f"{magnitudes[1]:.1f}"

    # A machine whose host blocks are three times more dispersed than every
    # other machine's is not measuring host cost, and its ratios set both ends
    # of the range if they are left in.
    dispersions = {
        collection: collection.dispersion("host_ns")
        for collection in newest_per_device(collections)
    }
    usable = [
        collection
        for collection, spread in dispersions.items()
        if spread is not None and spread <= HOST_DISPERSION_LIMIT
    ]
    excluded = [
        collection for collection in dispersions if collection not in usable
    ]

    host_ratios: list[tuple[float, str, str]] = []
    for collection in newest_per_device(collections):
        device = collection.devices.get("blade", "")
        slug = DEVICE_SLUG.get(device)
        if slug is None:
            print(f"warning: no macro slug for device {device!r}", file=sys.stderr)
            continue
        for metric, name in (("host_ns", "host"), ("gpu_ns", "gpu")):
            spread = collection.dispersion(metric)
            if spread is not None:
                values[f"dispersion/{slug}/{name}"] = f"{spread:.0f}"
        # How far the device moves during a measurement block. Negative means
        # it sped up: the clock was still ramping while it was being measured.
        drift = collection.drift.get("gpu_ns")
        if drift:
            values[f"drift/{slug}/median"] = f"{statistics.median(drift):+.1f}"
            values[f"drift/{slug}/worst"] = f"{max(drift, key=abs):+.1f}"
        for workload in WORKLOAD_ORDER:
            for metric, prefix in (
                ("gpu_ns", ""),
                ("host_ns", "host"),
                ("wait_ns", "wait"),
            ):
                # Metal does not supply a comparable command-buffer span:
                # Blade reports summed pass intervals and wgpu returned zero
                # for most workloads. Do not emit tempting `0 us` device
                # macros for a metric the paper explicitly withholds.
                if collection.manifest["backend"] == "metal" and metric == "gpu_ns":
                    continue
                baseline_us = process_median_us(
                    collection, "blade", "automatic", workload, 16, metric
                )
                if baseline_us is None:
                    continue
                values[f"{slug}/{workload}/{prefix}autous"] = (
                    f"{baseline_us:.1f}"
                )
                for name, implementation, policy in COMPARISONS:
                    # Stale wgpu shaders invalidate what the GPU executed, so
                    # the device and wait metrics are withheld; the host-side
                    # recording cost is unaffected and stays quotable.
                    if implementation != "blade" and (
                        collection.mismatched_devices
                        or (collection.wgpu_shader_checks and metric != "host_ns")
                    ):
                        continue
                    contender_us = process_median_us(
                        collection, implementation, policy, workload, 16, metric
                    )
                    if contender_us is None:
                        continue
                    values[f"{slug}/{workload}/{prefix}{name}us"] = (
                        f"{contender_us:.1f}"
                    )
                    # Wait time is quoted as a median and never as an interval,
                    # so resampling it would cost a third of the run time to
                    # produce numbers nothing cites.
                    if prefix == "wait":
                        continue
                    result = comparison(
                        collection,
                        workload,
                        metric,
                        implementation,
                        policy,
                        bootstrap_samples,
                    )
                    if result is None:
                        continue
                    difference, low, high = result
                    record(f"{slug}/{workload}/{prefix}{name}", difference, low, high)
                    if name == "control" and not prefix:
                        values[f"{slug}/{workload}/floor"] = (
                            f"{max(abs(low), abs(high)):.1f}"
                        )
                    if prefix == "host" and name == "wgpu":
                        # `difference` is the median paired process-level
                        # percentage. Convert that same estimand to a ratio
                        # instead of taking a ratio of pooled sample medians.
                        ratio = 1.0 + difference / 100.0
                        host_ratios.append(
                            (ratio, slug, workload, collection in usable)
                        )

                # Scope while placement is manual is the direct check behind
                # the paper's destination-scope interpretation. It is not a
                # comparison against B-auto, so keep it out of COMPARISONS and
                # name its baseline explicitly.
                if metric == "gpu_ns":
                    manual_scope = comparison(
                        collection,
                        workload,
                        metric,
                        "blade",
                        "hazard-only-scoped",
                        bootstrap_samples,
                        against=("blade", "hazard-only"),
                    )
                    if manual_scope is not None:
                        difference, low, high = manual_scope
                        record(
                            f"{slug}/{workload}/manualscope",
                            difference,
                            low,
                            high,
                        )

    for suffix, selected in (
        ("", [r for r in host_ratios if r[3]]),
        ("/all", host_ratios),
    ):
        if not selected:
            continue
        values[f"host/ratio{suffix}/min"] = f"{min(selected)[0]:.1f}"
        values[f"host/ratio{suffix}/max"] = f"{max(selected)[0]:.1f}"
        values[f"host/ratio{suffix}/count"] = str(len(selected))
    if excluded:
        # No braces in a generated value: it is written inside the braces of a
        # `\csname` definition, and a `\texttt{...}` here truncates at the
        # first closing brace. The prose does the marking up.
        values["host/excluded/count"] = str(len(excluded))
        values["host/excluded/machines"] = ", ".join(
            collection.display_label for collection in excluded
        )
        values["host/excluded/dispersion"] = ", ".join(
            f"{dispersions[collection]:.0f}" for collection in excluded
        )
    if usable:
        values["host/usable/dispersion"] = (
            f"{max(dispersions[collection] for collection in usable):.0f}"
        )

    # Control floors, and how many cells each conclusion may be drawn from.
    floors: list[float] = []
    for collection in newest_per_device(collections):
        if not collection.has_scope_axis:
            continue
        for workload in WORKLOAD_ORDER:
            floor = control_floor(collection, workload, bootstrap_samples)
            if floor is not None:
                floors.append(floor)
    if floors:
        values["floor/count"] = str(len(floors))
        values["floor/max"] = f"{max(floors):.1f}"
        values["floor/above2"] = str(sum(1 for floor in floors if floor > 2.0))
        values["floor/above30"] = str(sum(1 for floor in floors if floor > 30.0))
        below = [floor for floor in floors if floor <= 2.0]
        values["floor/below2max"] = f"{max(below):.1f}" if below else "---"

    # The chain variants differ only by an initial barrier that lies before the
    # first timestamp. Their timed barrier calls are therefore the same, and
    # this count is a measurement-validity check rather than an estimate of the
    # cost of a required barrier.
    chain_effects = []
    for collection in newest_per_device(collections):
        for workload in WORKLOAD_ORDER:
            if not workload.endswith("-chain"):
                continue
            floor = control_floor(collection, workload, bootstrap_samples)
            result = comparison(
                collection,
                workload,
                "gpu_ns",
                "blade",
                "hazard-only",
                bootstrap_samples,
            )
            if floor is None or result is None:
                continue
            difference, _, _ = result
            if clears_stability_floor(result, floor):
                chain_effects.append(abs(difference))
    values["chain/resolved"] = str(len(chain_effects))
    if chain_effects:
        values["chain/maxresolved"] = f"{max(chain_effects):.1f}"

    # Metal is reported as a ratio between the two implementations rather than
    # as a difference from B-auto, because only one policy exists there.
    metal = next(
        (
            collection
            for collection in newest_per_device(collections)
            if collection.manifest["backend"] == "metal"
        ),
        None,
    )
    if metal is not None:
        host_shares, wait_savings, metal_hosts = [], [], []
        for workload in WORKLOAD_ORDER:
            for metric, sink in (("host_ns", host_shares), ("wait_ns", wait_savings)):
                result = comparison(
                    metal,
                    workload,
                    metric,
                    "wgpu",
                    "tracked",
                    bootstrap_samples,
                )
                if result is not None:
                    difference, _, _ = result
                    sink.append(100.0 / (1.0 + difference / 100.0))
            blade_host = process_median_us(
                metal, "blade", "automatic", workload, 16, "host_ns"
            )
            if blade_host is not None:
                metal_hosts.append(blade_host)
        if host_shares:
            values["metal/hostshare/min"] = f"{min(host_shares):.0f}"
            values["metal/hostshare/max"] = f"{max(host_shares):.0f}"
        if wait_savings:
            values["metal/waitsaved/min"] = f"{100 - max(wait_savings):.0f}"
            values["metal/waitsaved/max"] = f"{100 - min(wait_savings):.0f}"
        if metal_hosts:
            values["metal/host/min"] = f"{min(metal_hosts):.0f}"
            values["metal/host/max"] = f"{max(metal_hosts):.0f}"
            vulkan = [
                value
                for collection in newest_per_device(collections)
                if DEVICE_SLUG.get(collection.devices.get("blade", "")) == "rtx5070"
                for w in ("compute-independent", "graphics-independent")
                if (
                    value := process_median_us(
                        collection, "blade", "automatic", w, 16, "host_ns"
                    )
                )
                is not None
            ]
            if vulkan:
                ratios = sorted(
                    (
                        min(metal_hosts) / min(vulkan),
                        max(metal_hosts) / max(vulkan),
                    )
                )
                values["metal/overvulkan/min"] = f"{ratios[0]:.1f}"
                values["metal/overvulkan/max"] = f"{ratios[1]:.1f}"

    # Overlap-depth effects for every swept device, cited by pass count. The
    # prose reads the growth-and-saturation of the Radeon 780M penalty from
    # these rather than re-deriving it from the figure by eye.
    for collection in newest_sweeps(collections, "sweep-gpu"):
        slug = DEVICE_SLUG.get(collection.devices.get("blade", ""))
        if slug is None:
            continue
        for count in [c for c in collection.pass_counts() if c <= 64]:
            for name, implementation, policy in (
                ("placement", "blade", "hazard-only"),
                ("wgpu", "wgpu", "tracked"),
            ):
                result = comparison(
                    collection,
                    "graphics-independent",
                    "gpu_ns",
                    implementation,
                    policy,
                    bootstrap_samples,
                    passes=count,
                )
                if result is not None:
                    record(
                        f"depth/{slug}/graphics-independent/p{count}/{name}",
                        *result,
                    )

    values.update(launch_state_numbers(collections))
    values.update(sweep_numbers(collections))
    values.update(profile_numbers(raw))
    values.update(study_counts(raw))

    # Prose fragments that exist only when the data calls for them. A machine
    # that measures cleanly should appear in the paper as an ordinary platform,
    # with no paragraph explaining itself; one that does not has to be excluded
    # in the text, not silently. Generating the sentence from the same check
    # that excludes the data keeps the two in step.
    if excluded:
        names = ", ".join(
            f"\\texttt{{{collection.display_label}}}" for collection in excluded
        )
        caveat = (
            f"\\textbf{{Host measurements on {names}.}} These are reported in "
            "Table~\\ref{tab:host-matrix} and used for nothing. At the median "
            "16-pass block those host samples spread "
            "$\\bladenum{host/excluded/dispersion}\\%$ of their own median, "
            "against at most $\\bladenum{host/usable/dispersion}\\%$ elsewhere, "
            "while the same collection's \\emph{device} blocks are as clean as "
            "anyone's. That is CPU-side interference confined to individual "
            "blocks, which is what a control floor cannot see, since both "
            "control configurations sit inside the same block."
        )
        inline = (
            f" The {names} row is shown but not used, for the reason given in "
            "Section~\\ref{sec:deviations}."
        )
    else:
        caveat = ""
        inline = ""
    prose = [
        f"\\newcommand{{\\hostcaveat}}{{{caveat}}}",
        f"\\newcommand{{\\hostcaveatinline}}{{{inline}}}",
    ]

    lines = [NUMBERS_PREAMBLE, *prose]
    for key in sorted(values):
        if "{" in values[key] or "}" in values[key]:
            raise ValueError(
                f"generated value for {key!r} contains a brace, which "
                f"truncates the definition: {values[key]!r}"
            )
        lines.append(
            f"\\expandafter\\gdef\\csname blade@num@{key}\\endcsname{{{values[key]}}}"
        )
    return "\n".join(lines) + "\n"


def marginal_cost(
    collection: Collection,
    implementation: str,
    policy: str,
    workload: str,
    metric: str,
    cap: int,
) -> float | None:
    """Average process-median increment per pass between the endpoint counts."""
    counts = [count for count in collection.pass_counts() if count <= cap]
    if len(counts) < 2:
        return None
    first = process_median_us(
        collection, implementation, policy, workload, counts[0], metric
    )
    last = process_median_us(
        collection, implementation, policy, workload, counts[-1], metric
    )
    if first is None or last is None:
        return None
    return (last - first) / (counts[-1] - counts[0])


def sweep_numbers(collections: list[Collection]) -> dict[str, str]:
    """Marginal cost per pass and the value at sixteen passes, from the sweeps.

    Both sides come from `sweep_pair`, so a host number and the device number
    beside it always describe the same machine and device.
    """
    pair = sweep_pair(collections)
    if pair is None:
        return {}
    gpu, cpu = pair
    sources = {
        "host": (cpu, "host_ns", HOST_MARGINAL_CAP),
        "gpu": (gpu, "gpu_ns", 64),
    }
    values: dict[str, str] = {}
    for side, (collection, metric, cap) in sources.items():
        if collection is None:
            continue
        for workload in ("compute-independent", "graphics-independent"):
            for name, implementation, policy in (
                ("auto", "blade", "automatic"),
                ("hazard", "blade", "hazard-only"),
                ("wgpu", "wgpu", "tracked"),
            ):
                at16 = process_median_us(
                    collection, implementation, policy, workload, 16, metric
                )
                key = f"sweep/{side}/{workload}/{name}"
                marginal = marginal_cost(
                    collection, implementation, policy, workload, metric, cap
                )
                if marginal is not None:
                    values[f"{key}/marginal"] = f"{marginal:.2f}"
                if at16 is not None:
                    values[f"{key}/at16"] = f"{at16:.1f}"
    # Ratios the prose quotes directly, so they cannot be recomputed wrongly.
    for workload in ("compute-independent", "graphics-independent"):
        base = f"sweep/host/{workload}"
        for reference in ("auto", "hazard"):
            numerator = values.get(f"{base}/wgpu/marginal")
            denominator = values.get(f"{base}/{reference}/marginal")
            if numerator and denominator and float(denominator):
                values[f"{base}/wgpuover{reference}"] = (
                    f"{float(numerator) / float(denominator):.1f}"
                )
        auto = values.get(f"{base}/auto/marginal")
        hazard = values.get(f"{base}/hazard/marginal")
        if auto and hazard:
            values[f"{base}/barriercost"] = f"{float(auto) - float(hazard):.2f}"
    for workload in ("compute-independent", "graphics-independent"):
        base = f"sweep/gpu/{workload}"
        auto = values.get(f"{base}/auto/marginal")
        hazard = values.get(f"{base}/hazard/marginal")
        if auto and hazard:
            values[f"{base}/barriercost"] = f"{float(auto) - float(hazard):.2f}"
    return values


def study_counts(raw: Path) -> dict[str, str]:
    """How much was run, and whether the runs agreed with each other.

    The paper states these as evidence of coverage, so they are counted from
    the collections rather than remembered. Every one of them was wrong by the
    time the fourth machine reported.
    """
    values: dict[str, str] = {}
    hashes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    matrix_runs = sweep_runs = 0
    for directory in sorted(raw.iterdir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != "blade-sync-study-v1":
            continue
        if manifest.get("parameters", {}).get("validation", False):
            # Correctness matrices use deliberately tiny shapes and must not
            # inflate either the performance-run count or its hash coverage.
            continue
        sweep = bool(manifest.get("parameters", {}).get("pass_list"))
        for run in manifest.get("runs", ()):
            if sweep:
                sweep_runs += 1
            else:
                matrix_runs += 1
            command = run["command"]
            workload = command[command.index("--workload") + 1]
            passes = (
                command[command.index("--passes") + 1]
                if "--passes" in command
                else "16"
            )
            digest = run["metadata"].get("validation_hash")
            if digest:
                hashes[directory.name, workload, passes].add(digest)
    values["runs/matrix"] = str(matrix_runs)
    values["runs/sweep"] = str(sweep_runs)
    values["runs/hashgroups"] = str(len(hashes))
    values["runs/hashconflicts"] = str(
        sum(1 for digests in hashes.values() if len(digests) > 1)
    )

    # Captures: this artifact extracts barrier records, not the complete
    # submitted command stream. Count how many machines produced the same
    # extracted table.
    newest_capture: dict[str, tuple[str, Path]] = {}
    for directory in sorted(raw.glob("*-captures")):
        barriers = directory / "barriers.csv"
        if not barriers.is_file():
            continue
        manifest = json.loads((directory / "manifest.json").read_text())
        host = str(manifest.get("host", ""))
        stamp = str(manifest.get("created_utc", ""))
        if host not in newest_capture or stamp > newest_capture[host][0]:
            newest_capture[host] = (stamp, directory)
    tables: dict[str, list[str]] = defaultdict(list)
    table_paths: dict[str, Path] = {}
    for host, (_, directory) in sorted(newest_capture.items()):
        barriers = directory / "barriers.csv"
        digest = hashlib.sha256(barriers.read_bytes()).hexdigest()
        tables[digest].append(host)
        table_paths[digest] = barriers
    if tables:
        # Row count and per-configuration counts come from the majority table,
        # not from whichever host happened to be read last.
        digest, hosts = max(tables.items(), key=lambda item: len(item[1]))
        rows = list(csv.DictReader(table_paths[digest].open()))
        values["captures/machines"] = str(len(hosts))
        values["captures/rows"] = str(len(rows))
        values["captures/sha"] = digest[:12]
        values["captures/distinct"] = str(len(tables))
        counted = defaultdict(set)
        for row in rows:
            counted[row["implementation"], row["policy"], row["workload"]].add(
                row["index"]
            )
        for implementation, policy, name in (
            ("blade", "automatic", "auto"),
            ("blade", "hazard-only", "hazard"),
            ("wgpu", "tracked", "wgpu"),
        ):
            for family, workload in (
                ("independent", "compute-independent"),
                ("chain", "compute-chain"),
            ):
                found = counted.get((implementation, policy, workload))
                if found:
                    values[f"barriers/{name}/{family}"] = str(len(found))

        # `after_work` is derived from Vulkan queue-submission order, not host
        # command-buffer recording order. wgpu records pass and transition
        # command buffers separately and interleaves them only at submit time,
        # so the distinction is essential for a placement statement.
        manifest = json.loads(
            (table_paths[digest].parent / "manifest.json").read_text()
        )
        shape = manifest.get("capture_shape", ())
        passes = 0
        if "--passes" in shape:
            passes = int(shape[shape.index("--passes") + 1])
        if passes and rows and "after_work" in rows[0]:
            for implementation, policy, name in (
                ("blade", "automatic", "auto"),
                ("blade", "hazard-only", "hazard"),
                ("wgpu", "tracked", "wgpu"),
            ):
                for family, workload in (
                    ("independent", "compute-independent"),
                    ("chain", "compute-chain"),
                ):
                    positions = [
                        int(row["after_work"])
                        for row in rows
                        if row["implementation"] == implementation
                        and row["policy"] == policy
                        and row["workload"] == workload
                    ]
                    if not positions:
                        continue
                    values[f"barriers/{name}/{family}/initial"] = str(
                        sum(position == 0 for position in positions)
                    )
                    values[f"barriers/{name}/{family}/interpass"] = str(
                        sum(0 < position < passes for position in positions)
                    )
                    values[f"barriers/{name}/{family}/final"] = str(
                        sum(position >= passes for position in positions)
                    )
    return values


def profile_numbers(raw: Path) -> dict[str, str]:
    """Component shares, and the aggregates the prose argues from."""
    values: dict[str, str] = {}
    spread: dict[tuple[str, str], list[float]] = {}
    for profile in newest_profile_per_host(raw):
        for (workload, implementation), shares in profile.shares.items():
            prefix = f"profile/{profile.host}/{implementation}/{workload}"
            for bucket, share in shares.items():
                slug = bucket.replace(" ", "").replace("/", "").replace("-", "")
                values[f"{prefix}/{slug}"] = f"{share:.1f}"
            wgpu_total = sum(
                share for bucket, share in shares.items() if bucket.startswith("wgpu")
            )
            values[f"{prefix}/wgputotal"] = f"{wgpu_total:.0f}"
            system = shares.get("kernel", 0.0) + shares.get("driver", 0.0)
            values[f"{prefix}/system"] = f"{system:.0f}"
            spread.setdefault((implementation, "system"), []).append(system)
            spread.setdefault((implementation, "wgputotal"), []).append(wgpu_total)
            tracker = shares.get("wgpu tracker")
            if tracker is not None:
                spread.setdefault((implementation, "tracker"), []).append(tracker)
    # Ranges across every profiled machine and workload, so the prose can quote
    # a span rather than pick one column and generalize from it.
    for (implementation, name), observed in spread.items():
        values[f"profile/{implementation}/{name}/min"] = f"{min(observed):.0f}"
        values[f"profile/{implementation}/{name}/max"] = f"{max(observed):.0f}"
    return values



# ---------------------------------------------------------------- figures

FIGURE_PREAMBLE = "% Generated by paper/build-tables.py -- do not edit by hand.\n"


def effect_figure(
    collections: list[Collection],
    bootstrap_samples: int,
    policy: str,
    caption: str,
    label: str,
) -> str | None:
    """One panel per device: effect per workload, with interval and floor.

    A bar is the percent difference from `B-auto`, the two black ticks are its
    95% interval, and the wider grey ticks are plus and minus that cell's
    post-hoc control threshold. A directional effect is read only when the
    whole interval lies beyond the corresponding grey tick.

    Interval and floor are drawn as marks rather than pgfplots error bars,
    which cannot take a numeric offset against a symbolic axis.
    """
    devices = [c for c in newest_per_device(collections) if c.has_scope_axis]
    if not devices:
        return None
    panels = []
    for collection in devices:
        device = DEVICE_SHORT.get(
            collection.devices.get("blade", ""), collection.devices.get("blade", "?")
        )
        bars, intervals, floors = [], [], []
        for workload in WORKLOAD_ORDER:
            result = comparison(
                collection, workload, "gpu_ns", "blade", policy, bootstrap_samples
            )
            floor = control_floor(collection, workload, bootstrap_samples)
            if result is None or floor is None:
                continue
            difference, low, high = result
            short = WORKLOAD_SHORT[workload]
            bars.append(f"({difference:.1f},{short})")
            intervals.append(f"({low:.1f},{short}) ({high:.1f},{short})")
            floors.append(f"({floor:.1f},{short}) ({-floor:.1f},{short})")
        if not bars:
            continue
        panels.append(
            "\\nextgroupplot[title={" + device + "}]\n"
            "\\addplot[xbar, fill=black!12, draw=black!55]\n"
            "coordinates {" + " ".join(bars) + "};\n"
            "\\addplot[only marks, mark=|, mark size=3pt, black, forget plot]\n"
            "coordinates {" + " ".join(intervals) + "};\n"
            "\\addplot[only marks, mark=|, mark size=5.5pt, black!35, forget plot]\n"
            "coordinates {" + " ".join(floors) + "};"
        )
    if not panels:
        return None
    body = "\n".join(panels)
    rows = -(-len(panels) // 3)
    workloads = ",".join(WORKLOAD_SHORT[w] for w in WORKLOAD_ORDER)
    return (
        FIGURE_PREAMBLE
        + "\\begin{figure*}[t]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{groupplot}[\n"
        f"  group style={{group size=3 by {rows}, horizontal sep=1.15cm,\n"
        "    vertical sep=1.75cm, y descriptions at=edge left},\n"
        "  width=5.2cm, height=4.3cm,\n"
        "  xbar, /pgf/bar width=5pt,\n"
        f"  symbolic y coords={{{workloads}}},\n"
        "  ytick=data, y dir=reverse,\n"
        "  xlabel={\\% of \\texttt{B-auto}},\n"
        "  xlabel style={font=\\scriptsize},\n"
        "  tick label style={font=\\scriptsize},\n"
        "  title style={font=\\scriptsize},\n"
        "  grid=major, grid style={black!10},\n"
        "  scaled x ticks=false,\n"
        "]\n"
        f"{body}\n"
        "\\end{groupplot}\n\\end{tikzpicture}\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure*}}\n"
    )


def marginal_figure(collections: list[Collection]) -> str | None:
    """Marginal cost of one additional pass, host and device side."""
    pair = sweep_pair(collections)
    if pair is None:
        return None
    device = DEVICE_SHORT.get(
        pair[0].devices.get("blade", ""), pair[0].devices.get("blade", "?")
    ).replace(" ", "~")
    values = sweep_numbers(collections)
    configurations = (("B-auto", "auto"), ("B-hazard", "hazard"), ("W-wgpu", "wgpu"))
    categories: list[str] = []
    coordinates: dict[str, list[str]] = {name: [] for name, _ in configurations}
    for side, side_label in (("host", "host"), ("gpu", "GPU")):
        for workload in ("compute-independent", "graphics-independent"):
            category = f"{side_label}~{WORKLOAD_SHORT[workload]}"
            present = False
            for name, config in configurations:
                value = values.get(f"sweep/{side}/{workload}/{config}/marginal")
                if value is None:
                    continue
                coordinates[name].append(f"({category},{float(value):.2f})")
                present = True
            if present:
                categories.append(category)
    if not categories:
        return None
    plots = "\n".join(
        "\\addplot+[ybar] coordinates {" + " ".join(coordinates[name]) + "};"
        for name, _ in configurations
        if coordinates[name]
    )
    legend = ",".join(
        f"\\texttt{{{name}}}" for name, _ in configurations if coordinates[name]
    )
    return (
        FIGURE_PREAMBLE
        + "\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n\\begin{axis}[\n"
        "  width=\\columnwidth, height=5.2cm,\n"
        "  ybar, /pgf/bar width=7pt,\n"
        f"  symbolic x coords={{{','.join(categories)}}},\n"
        "  xtick=data, ymin=0,\n"
        "  ylabel={$\\mu$s per additional pass},\n"
        "  ylabel style={font=\\scriptsize},\n"
        "  tick label style={font=\\scriptsize},\n"
        "  legend style={font=\\scriptsize, at={(0.5,1.03)}, anchor=south,\n"
        "    legend columns=3, draw=none},\n"
        "  grid=major, grid style={black!10},\n"
        "  nodes near coords,\n"
        "  every node near coord/.append style={font=\\tiny},\n"
        "]\n"
        f"{plots}\n"
        f"\\legend{{{legend}}}\n"
        "\\end{axis}\n\\end{tikzpicture}\n"
        f"\\caption{{Average cost of one additional pass on the {device}, from the "
        "pass-count sweeps. Host figures come from the timestamp-free collection "
        "and device figures from the GPU-timed one; each is the endpoint increment "
        "over the range stated in Table~\\ref{tab:sweep}. The gap between "
        "\\texttt{B-auto} and \\texttt{B-hazard} is what one redundant barrier "
        "costs; the gap to \\texttt{W-wgpu} is end-to-end.}\n"
        "\\label{fig:marginal}\n\\end{figure}\n"
    )


def paired_block_effects(
    collection: Collection,
    workload: str,
    passes: int,
    implementation: str,
    policy: str,
    metric: str = "gpu_ns",
) -> list[float]:
    """Per-repetition paired percent effects against `B-auto` at one count.

    These are the raw points behind the hierarchical estimate: one number per
    randomized block. The overlap-depth figure draws them individually because
    on one device they are bimodal, and a median with an interval would state
    that fact less plainly than the points themselves.
    """
    base = collection.blocked_values("blade", "automatic", workload, passes, metric)
    contender = collection.blocked_values(
        implementation, policy, workload, passes, metric
    )
    effects = []
    for repetition in sorted(set(base) & set(contender), key=str):
        baseline = statistics.median(base[repetition])
        if baseline:
            effects.append(
                100.0
                * (statistics.median(contender[repetition]) - baseline)
                / baseline
            )
    return effects


def overlap_figure(
    collections: list[Collection], bootstrap_samples: int
) -> str | None:
    """Placement and tracked effects against overlap depth, per swept device.

    One panel per GPU-timed sweep: the paired process-level effect of
    `B-hazard` and `W-wgpu` against `B-auto` on `graphics-independent`, at
    every measured pass count. Small marks are individual block effects; the
    line joins the medians. The block marks are the honest part: where a
    device assigns whole launches to different performance states, the
    mixture is visible rather than averaged away.
    """
    sweeps = newest_sweeps(collections, "sweep-gpu")
    workload = "graphics-independent"
    panels = []
    for collection in sweeps:
        device = DEVICE_SHORT.get(
            collection.devices.get("blade", ""), collection.devices.get("blade", "?")
        )
        counts = [count for count in collection.pass_counts() if count <= 64]
        series = []
        for policy_label, implementation, policy, line_style, dot_style in (
            ("B-hazard", "blade", "hazard-only", "black, mark=*, mark size=1.4pt",
             "black!35"),
            ("W-wgpu", "wgpu", "tracked",
             "black!55, densely dashed, mark=o, mark size=1.4pt", "black!20"),
        ):
            line, dots = [], []
            for count in counts:
                result = comparison(
                    collection,
                    workload,
                    "gpu_ns",
                    implementation,
                    policy,
                    bootstrap_samples,
                    passes=count,
                )
                if result is None:
                    continue
                line.append(f"({count},{result[0]:.1f})")
                dots.extend(
                    f"({count},{effect:.1f})"
                    for effect in paired_block_effects(
                        collection, workload, count, implementation, policy
                    )
                )
            if line:
                series.append(
                    f"\\addplot[only marks, mark=*, mark size=0.8pt, {dot_style},"
                    " forget plot] coordinates {" + " ".join(dots) + "};\n"
                    f"\\addplot[{line_style}] coordinates {{"
                    + " ".join(line)
                    + "};\n"
                    f"\\addlegendentry{{\\texttt{{{policy_label}}}}}"
                )
        if series:
            panels.append(
                f"\\nextgroupplot[title={{{device}}}]\n" + "\n".join(series)
            )
    if not panels:
        return None
    counts_text = ",".join(
        str(count)
        for count in sorted(
            {c for s in sweeps for c in s.pass_counts() if c <= 64}
        )
    )
    return (
        FIGURE_PREAMBLE
        + "\\begin{figure*}[t]\n\\centering\n\\begin{tikzpicture}\n"
        "\\begin{groupplot}[\n"
        f"  group style={{group size={len(panels)} by 1, horizontal sep=1.4cm}},\n"
        "  width=8.0cm, height=5.4cm,\n"
        f"  symbolic x coords={{{counts_text}}},\n"
        "  xtick=data,\n"
        "  xlabel={passes per command buffer},\n"
        "  ylabel={\\% of \\texttt{B-auto}},\n"
        "  xlabel style={font=\\scriptsize},\n"
        "  ylabel style={font=\\scriptsize},\n"
        "  tick label style={font=\\scriptsize},\n"
        "  title style={font=\\scriptsize},\n"
        "  legend style={font=\\scriptsize, draw=none, at={(0.03,0.97)},\n"
        "    anchor=north west},\n"
        "  grid=major, grid style={black!10},\n"
        "]\n"
        + "\n".join(panels)
        + "\n\\end{groupplot}\n\\end{tikzpicture}\n"
        "\\caption{GPU-span effect of removing the redundant barriers "
        "(\\texttt{B-hazard}) and of the tracked implementation "
        "(\\texttt{W-wgpu}) against \\texttt{B-auto} on "
        "\\texttt{graphics-independent}, as the number of overlapping passes "
        "grows. Each small mark is one paired process launch; the line joins "
        "the medians of those block effects. The launch-level split on the "
        "Radeon~780M is the bimodality of Section~\\ref{sec:deviations}: "
        "whole launches land in one of two performance states, and the "
        "placement penalty reproduces inside each.}\n"
        "\\label{fig:overlap-depth}\n\\end{figure*}\n"
    )


def launch_state_numbers(collections: list[Collection]) -> dict[str, str]:
    """Launch-level performance-state statistics for the swept devices.

    A launch counts as slow when its block median exceeds 1.5 times the
    fastest block of the same configuration and pass count; the observed
    split is a factor of about two, so the threshold is a separator rather
    than a tuning knob. Keys are only emitted for a device where slow
    launches exist, so citing them for a device that has none is a build
    error rather than a claim.
    """
    values: dict[str, str] = {}
    workload = "graphics-independent"
    for collection in newest_sweeps(collections, "sweep-gpu"):
        slug = DEVICE_SLUG.get(collection.devices.get("blade", ""))
        if slug is None:
            continue
        counted = {"blade": [0, 0], "wgpu": [0, 0]}
        ratios: list[float] = []
        fast_modes: dict[tuple[str, str, int], float] = {}
        for key in sorted(collection.samples, key=str):
            implementation, policy, key_workload, passes = key
            if key_workload != workload:
                continue
            blocks = collection.blocked_values(
                implementation, policy, key_workload, passes, "gpu_ns"
            )
            medians = sorted(
                statistics.median(values_) for values_ in blocks.values()
            )
            if not medians:
                continue
            fastest = medians[0]
            fast = [m for m in medians if m <= 1.5 * fastest]
            slow = [m for m in medians if m > 1.5 * fastest]
            counted[implementation][0] += len(slow)
            counted[implementation][1] += len(medians)
            fast_modes[implementation, policy, passes] = statistics.median(fast)
            ratios.extend(m / statistics.median(fast) for m in slow)
        if not ratios:
            continue
        for implementation, (slow_count, total) in counted.items():
            values[f"bimodal/{slug}/{implementation}/slow"] = str(slow_count)
            values[f"bimodal/{slug}/{implementation}/launches"] = str(total)
        values[f"bimodal/{slug}/ratio"] = f"{statistics.median(ratios):.2f}"
        auto = fast_modes.get(("blade", "automatic", 16))
        hazard = fast_modes.get(("blade", "hazard-only", 16))
        if auto and hazard:
            values[f"bimodal/{slug}/fastplacement"] = (
                f"{100.0 * (hazard - auto) / auto:+.1f}"
            )
    return values


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
                "processes",
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
                process_count = len(collection.block_samples[key])
                medians = {
                    metric: process_median_us(
                        collection,
                        implementation,
                        policy,
                        workload,
                        passes,
                        metric,
                    )
                    for metric in ("host_ns", "gpu_ns", "wait_ns")
                }
                assert all(value is not None for value in medians.values())
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
                        process_count,
                        len(bucket["gpu_ns"]),
                        f"{medians['host_ns']:.3f}",
                        f"{medians['gpu_ns']:.3f}",
                        f"{medians['wait_ns']:.3f}",
                    )
                )


def main() -> None:
    arguments = parse_arguments()
    if arguments.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    arguments.output.mkdir(parents=True, exist_ok=True)
    collections = discover_collections(arguments.raw)
    if not collections:
        raise ValueError(f"no collections found under {arguments.raw}")

    # The prose names devices; the tables are generated. If a machine's raw
    # data is not on this disk the tables lose its rows silently while the text
    # keeps citing them, which is the one failure mode of keeping measurements
    # out of git.
    present = {
        DEVICE_SHORT.get(name, name)
        for collection in newest_per_device(collections)
        for name in collection.devices.values()
    }
    cited = set()
    paper = HERE / "main.tex"
    if paper.is_file():
        text = paper.read_text()
        for device in DEVICE_SHORT.values():
            if device.replace(" ", "~") in text or device in text:
                cited.add(device)
    missing = sorted(cited - present)
    if missing:
        message = (
            "main.tex cites devices with no matrix data under data/raw:\n  "
            + "\n  ".join(missing)
            + "\n  Their table rows will be absent while the text still refers "
            "to them.\n  Copy those collections back before building the PDF."
        )
        if not arguments.allow_incomplete:
            raise SystemExit(
                message
                + "\nUse --allow-incomplete only to generate partial audit outputs."
            )
        print("WARNING: " + message, file=sys.stderr)

    stale = [
        collection
        for collection in newest_per_device(collections)
        if collection.wgpu_shader_checks and "wgpu" in collection.devices
    ]
    if stale:
        print(
            "WARNING: these collections ran wgpu with its injected shader "
            "checks, so their W-wgpu cells measure code generation as much as "
            "synchronization:\n  "
            + "\n  ".join(
                f"{collection.root.name} ({collection.display_label})"
                for collection in stale
            )
            + "\n  Those cells are withheld and their prose macros are not "
            "emitted, so main.tex will fail to build wherever it cites one. "
            "Recollect with a wgpu at or after the shader-parity commit.",
            file=sys.stderr,
        )

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
        "fig-placement.tex": effect_figure(
            collections,
            arguments.bootstrap_samples,
            "hazard-only",
            "Barrier placement: percent difference in GPU span between "
            "\\texttt{B-hazard} and \\texttt{B-auto} at 16 passes, by device and "
            "workload. The bar is the median paired process effect, the two black "
            "ticks its 95\\% hierarchical-bootstrap interval, and the wider grey ticks plus and minus that "
            "cell's post-hoc stability threshold. A directional effect is read only when its "
            "whole interval lies beyond the corresponding grey tick. Note the "
            "per-panel horizontal scales.",
            "fig:placement",
        ),
        "fig-scope.tex": effect_figure(
            collections,
            arguments.bootstrap_samples,
            "automatic-scoped",
            "Barrier scope at fixed placement: percent difference in GPU span "
            "between \\texttt{B-auto-scoped} and \\texttt{B-auto}, drawn as in "
            "Figure~\\ref{fig:placement}. The chain workloads are the clean test, "
            "because every timed inter-pass barrier is retained there. The grey "
            "threshold comes from the global identical-request control; it is a "
            "cell-stability diagnostic, not a scoped control.",
            "fig:scope",
        ),
        "fig-marginal.tex": marginal_figure(collections),
        "fig-depth.tex": overlap_figure(collections, arguments.bootstrap_samples),
        "profile.tex": build_profile_table(arguments.raw),
        "numbers.tex": numbers_macros(
            collections, arguments.raw, arguments.bootstrap_samples
        ),
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

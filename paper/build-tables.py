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
    return MACHINE_LABEL.get(host, host)


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
        manifest = root / "manifest.json"
        if not manifest.is_file():
            continue
        schema = json.loads(manifest.read_text()).get("schema", "")
        if schema != "blade-sync-study-v1":
            # Profiles and captures live alongside the timing collections but
            # are read by their own table builders.
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
        # Per-block clock drift, keyed by metric. One block is one process
        # launch, so the device idles between blocks and ramps again inside
        # each; pooling the samples hides that, which is why it is measured
        # here while the file is still in order.
        self.drift: dict[str, list[float]] = defaultdict(list)
        for path in sorted(root.glob("r*.csv")):
            run = analyze.read_run(path)
            implementation = run.metadata["implementation"]
            self.devices[implementation] = run.metadata["device_name"]
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
        for key, bucket in self.samples.items():
            if key[3] != 16:
                continue
            values = sorted(bucket.get(metric, []))
            if len(values) < 4:
                continue
            quartiles = statistics.quantiles(values, n=4)
            median = statistics.median(values)
            if median:
                spreads.append(100.0 * (quartiles[2] - quartiles[0]) / median)
        return statistics.median(spreads) if spreads else None


def median_us(values: Sequence[float]) -> float:
    return statistics.median(values) / 1000.0


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
) -> tuple[float, float, float] | None:
    """Percent difference from `against`, with a bootstrap 95% interval.

    The seed is derived here rather than at each call site, so every table and
    every number in the prose that names the same comparison gets the same
    resampling and therefore the same interval. Two builders passing seeds of
    different shapes for one quantity is how a table and the sentence beside it
    come to disagree in the last digit.
    """
    key = (collection.root.name, workload, metric, implementation, policy, against)
    if key in _INTERVALS:
        return _INTERVALS[key]
    baseline = collection.values(*against, workload, 16, metric)
    contender = collection.values(implementation, policy, workload, 16, metric)
    if not baseline or not contender:
        return None
    result = relative_difference(
        baseline, contender, bootstrap_samples, tuple(str(part) for part in key[:5])
    )
    _INTERVALS[key] = result
    return result


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
    preamble_rows: Sequence[str] = (),
) -> str:
    environment = "table*" if star else "table"
    lines = [
        f"% Generated by paper/build-tables.py -- do not edit by hand.",
        f"\\begin{{{environment}}}[t]",
        "\\centering",
        "\\small",
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
            "Machines were collected over two days, so they do not all sit on "
            "one commit. The measured code --- \\texttt{blade-graphics} and the "
            "benchmark --- is identical across every Vulkan revision listed "
            "except for the addition of an inert \\texttt{--capture} flag; the "
            "Metal row predates the scope axis, which is why it has no scoped "
            "configurations."
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
    withheld = [c for c in newest_per_device(collections) if c.mismatched_devices]
    return latex_table(
        caption=(
            "GPU span of one 16-pass command buffer on Vulkan: \\texttt{B-auto} median "
            "in microseconds, then percent differences from it with 95\\% bootstrap "
            "intervals. \\texttt{B-explicit-all} is the instrumentation control: it "
            "emits the same commands as \\texttt{B-auto}, so where it is not "
            "indistinguishable the cell is not measuring. On the Intel part it is "
            "not, in four cells, and nothing is read from those "
            "(Section~\\ref{sec:deviations})."
        ),
        label="tab:gpu-matrix",
        column_spec="@{}llr" + "r@{\\,}l" * 3 + "@{}",
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
            "median microseconds, then percent differences from \\texttt{B-auto} with "
            "95\\% bootstrap intervals. These collections have GPU timestamps enabled, "
            "which costs Blade one \\texttt{vkCmdWriteTimestamp} per pass against two "
            "in total for wgpu, so the comparison is conservative for Blade."
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
    collection = next(
        (c for c in collections if c.manifest["backend"] == "metal"), None
    )
    if collection is None:
        return None
    body = []
    for workload in WORKLOAD_ORDER:
        row = [WORKLOAD_SHORT[workload]]
        measured = False
        for implementation, policy in (("blade", "automatic"), ("wgpu", "tracked")):
            for metric in ("host_ns", "wait_ns"):
                values = collection.values(
                    implementation, policy, workload, 16, metric
                )
                row.append(f"{median_us(values):.1f}" if values else "---")
                measured = measured or bool(values)
        # The Metal collection predates the mixed workloads, so their rows
        # would be four dashes claiming a measurement was attempted.
        if measured:
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
        column_spec="@{}lrrrr@{}",
        header=("Workload", "host", "wait", "host", "wait"),
        preamble_rows=(
            " & \\multicolumn{2}{c}{\\blade{}} & \\multicolumn{2}{c}{\\wgpu{}} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
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
            "Pass-count sweep on the RTX~5070, medians in microseconds. "
            "Host rows come from the timestamp-free collection; GPU rows come from the "
            "GPU-timed collection. The last column is the marginal cost per additional "
            "pass over the measured range."
        ),
        label="tab:sweep",
        column_spec="@{}ll" + "r" * len(passes) + "r@{}",
        header=("", "Config", *(str(count) for count in passes), "$\\mu$s/pass"),
        body=body,
        star=True,
        note=(
            "Both cost components are linear in the pass count over this range, "
            "which is what makes a single marginal cost meaningful."
        ),
    )


def control_floor(
    collection: Collection, workload: str, bootstrap_samples: int
) -> float | None:
    """The largest disagreement between two identical command streams that one
    cell is consistent with.

    `explicit-all` and `automatic` emit the same commands, so any difference
    between them is measurement error, and an effect smaller than that error
    means nothing. The floor is the far end of the control's interval rather
    than its point estimate: on the Intel part's `compute-independent` cell the
    two agree to 2.8% and the interval reaches 71%, because the samples there
    are bimodal between two clock states and the median lands wherever the
    split falls. A floor of 2.8% would license a 14% reading from that cell;
    the interval says the cell cannot resolve 14%.

    It is computed per workload rather than per device because noise is not a
    property of the device alone: on the RX 7900 XT one compute cell resolves
    to 6% and the other to 0.1%, and a per-device maximum would throw away the
    half that works.
    """
    result = comparison(
        collection, workload, "gpu_ns", "blade", "explicit-all", bootstrap_samples
    )
    if result is None:
        return None
    _, low, high = result
    return max(abs(low), abs(high))


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
        if index:
            rows.append([RULE])
        first = True
        for workload in WORKLOAD_ORDER:
            baseline = collection.values("blade", "automatic", workload, 16, "gpu_ns")
            if not baseline:
                continue
            floor = control_floor(collection, workload, bootstrap_samples)
            cells = [
                device if first else "",
                WORKLOAD_SHORT[workload],
                f"{median_us(baseline):.1f}",
                f"{floor:.1f}" if floor is not None else "---",
            ]
            first = False
            for policy in ("automatic-scoped", "hazard-only", "hazard-only-scoped"):
                result = comparison(
                    collection, workload, "gpu_ns", "blade", policy, bootstrap_samples
                )
                if result is None:
                    cells.extend(["---", ""])
                    continue
                difference, low, high = result
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
        " ``ctrl'' is the control floor of that cell: the far end of the "
        "interval on the disagreement between \\texttt{explicit-all} and "
        "\\texttt{automatic} at global scope. Those two emit identical "
        "commands, so it is the smallest effect the cell can resolve, and a "
        "difference below it means nothing however tight its own interval "
        "looks. It is per cell rather than per device because noise is not a "
        "property of the device alone."
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
        column_spec="@{}llrr" + "r@{\\,}l" * 3 + "@{}",
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
    def driver_library(self) -> str:
        """The shared object that supplied the `driver` bucket.

        The profiling step does not pin an adapter, so a multi-GPU machine's
        columns cannot be labelled with a device. The driver can still be named,
        because it is the object the samples landed in, and for the one row
        whose value depends on the vendor that is the identification that
        matters.
        """
        weights: dict[str, float] = defaultdict(float)
        for row in csv.DictReader((self.root / "symbols.csv").open()):
            if row["bucket"] == "driver":
                weights[row["dso"]] += float(row["self_percent"])
        return max(weights, key=weights.get, default="")


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
    profiles = [
        Profile(directory)
        for directory in sorted(raw.glob("*-profile"))
        if (directory / "buckets.csv").is_file()
        and (directory / "manifest.json").is_file()
    ]
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

    spanning = [""]
    index = 2
    rules = []
    for profile in profiles:
        width = 2 * len(profile.workloads)
        spanning.append(
            f"\\multicolumn{{{width}}}{{c}}{{\\texttt{{{profile.host}}}}}"
        )
        rules.append(f"\\cmidrule(lr){{{index}-{index + width - 1}}}")
        index += width
    header = ["Component"] + [
        f"\\textsc{{{implementation[0]}}} {WORKLOAD_SHORT.get(workload, workload)}"
        for _, workload, implementation in columns
    ]
    drivers = ", ".join(
        f"\\texttt{{{profile.host}}} on \\texttt{{{profile.driver_library}}}"
        for profile in profiles
        if profile.driver_library
    ).replace("_", "\\_")
    return latex_table(
        caption=(
            "Share of process CPU time by component, from flat \\texttt{perf} "
            "profiles. \\textsc{b} is \\blade{} and \\textsc{w} the matched "
            "\\wgpu{} program. The profiled workload is shaped for the host --- "
            "tiny dispatches, small targets, many passes, no timestamp queries "
            "--- so the process stays in the recording path."
        ),
        label="tab:profile",
        column_spec="@{}l" + "r" * len(columns) + "@{}",
        header=header,
        body=body,
        star=True,
        preamble_rows=(" & ".join(spanning) + " \\\\", "".join(rules)),
        note=(
            "Shares within a process, not times between processes: "
            "\\texttt{task-clock} charges a blocking fence wait to the process "
            "whenever the driver spins inside it, which inflates the "
            "\\emph{driver} row differently for the two implementations. Self "
            "time is attributed to the symbol a sample landed in, so inlined "
            "tracker work is charged to its caller and the tracker row is a "
            "lower bound. Columns sum to slightly more or less than 100 because "
            "\\texttt{perf} rounds each symbol's share independently. "
            f"The \\emph{{driver}} row is {drivers}. Unlike the timing "
            "collections, the profiling step does not pin an adapter, so these "
            "columns are labelled by machine and driver rather than by device; "
            "only shares within one process are read from them."
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
% 95\% bootstrap interval, signed and unsigned to match the two above.
\newcommand{\bpctci}[3]{\bpct{#1}{#2}{#3}\,[\bladenum{#1/#2/#3/lo},\,\bladenum{#1/#2/#3/hi}]}
\newcommand{\bmagci}[3]{\bmag{#1}{#2}{#3}\,[\bladenum{#1/#2/#3/maglo},\,\bladenum{#1/#2/#3/maghi}]}
% The control floor of one cell: the largest disagreement between two
% configurations emitting identical commands that the cell is consistent with.
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
        values[f"{key}/mag"] = f"{abs(difference):.1f}"
        # The unsigned form says "worth 26.3% [20.1, 28.0]", which is only a
        # sentence one may write when the interval has a sign. Leaving these
        # keys undefined for a cell whose interval spans zero makes `\magci`
        # fail the build there rather than print a nonsense range.
        if (low > 0) == (high > 0):
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
            values[f"drift/{slug}/worst"] = f"{min(drift):+.1f}"
        for workload in WORKLOAD_ORDER:
            for metric, prefix in (
                ("gpu_ns", ""),
                ("host_ns", "host"),
                ("wait_ns", "wait"),
            ):
                baseline = collection.values(
                    "blade", "automatic", workload, 16, metric
                )
                if not baseline:
                    continue
                values[f"{slug}/{workload}/{prefix}autous"] = (
                    f"{median_us(baseline):.1f}"
                )
                for name, implementation, policy in COMPARISONS:
                    if implementation != "blade" and collection.mismatched_devices:
                        continue
                    contender = collection.values(
                        implementation, policy, workload, 16, metric
                    )
                    if not contender:
                        continue
                    values[f"{slug}/{workload}/{prefix}{name}us"] = (
                        f"{median_us(contender):.1f}"
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
                        ratio = median_us(contender) / median_us(baseline)
                        host_ratios.append(
                            (ratio, slug, workload, collection in usable)
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

    # The largest placement effect anywhere on a dependent chain, where the
    # redundant barrier is the only thing placement can remove. The claim that
    # a needed barrier is free is exactly the claim that this stays small.
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
            difference, low, high = result
            resolved = (low > 0) == (high > 0) and abs(difference) > floor
            if resolved:
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
                blade = metal.values("blade", "automatic", workload, 16, metric)
                tracked = metal.values("wgpu", "tracked", workload, 16, metric)
                if blade and tracked:
                    sink.append(100.0 * median_us(blade) / median_us(tracked))
            blade_host = metal.values("blade", "automatic", workload, 16, "host_ns")
            if blade_host:
                metal_hosts.append(median_us(blade_host))
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
                median_us(collection.values("blade", "automatic", w, 16, "host_ns"))
                for collection in newest_per_device(collections)
                if DEVICE_SLUG.get(collection.devices.get("blade", "")) == "rtx5070"
                for w in ("compute-independent", "graphics-independent")
                if collection.values("blade", "automatic", w, 16, "host_ns")
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


def sweep_numbers(collections: list[Collection]) -> dict[str, str]:
    """Marginal cost per pass and the value at sixteen passes, from the sweeps."""
    sources = {
        "host": (next((c for c in collections if c.role == "sweep-cpu"), None), "host_ns"),
        "gpu": (next((c for c in collections if c.role == "sweep-gpu"), None), "gpu_ns"),
    }
    values: dict[str, str] = {}
    for side, (collection, metric) in sources.items():
        if collection is None:
            continue
        passes = [count for count in collection.pass_counts() if count <= 64]
        if not passes:
            continue
        for workload in ("compute-independent", "graphics-independent"):
            for name, implementation, policy in (
                ("auto", "blade", "automatic"),
                ("hazard", "blade", "hazard-only"),
                ("wgpu", "wgpu", "tracked"),
            ):
                first = collection.values(
                    implementation, policy, workload, passes[0], metric
                )
                last = collection.values(
                    implementation, policy, workload, passes[-1], metric
                )
                at16 = collection.values(implementation, policy, workload, 16, metric)
                key = f"sweep/{side}/{workload}/{name}"
                if first and last:
                    marginal = (median_us(last) - median_us(first)) / (
                        passes[-1] - passes[0]
                    )
                    values[f"{key}/marginal"] = f"{marginal:.2f}"
                if at16:
                    values[f"{key}/at16"] = f"{median_us(at16):.1f}"
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

    # Captures: the claim is that the submitted stream does not vary, so what
    # matters is how many machines and drivers produced the identical table.
    tables: dict[str, list[str]] = defaultdict(list)
    drivers: set[str] = set()
    rows: list[dict[str, str]] = []
    for directory in sorted(raw.glob("*-captures")):
        barriers = directory / "barriers.csv"
        if not barriers.is_file():
            continue
        digest = hashlib.sha256(barriers.read_bytes()).hexdigest()
        host = json.loads((directory / "manifest.json").read_text()).get("host", "")
        tables[digest].append(str(host))
        rows = list(csv.DictReader(barriers.open()))
    if tables:
        digest, hosts = max(tables.items(), key=lambda item: len(item[1]))
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
    return values


def profile_numbers(raw: Path) -> dict[str, str]:
    """Component shares, and the aggregates the prose argues from."""
    values: dict[str, str] = {}
    spread: dict[tuple[str, str], list[float]] = {}
    for directory in sorted(raw.glob("*-profile")):
        if not (directory / "buckets.csv").is_file():
            continue
        profile = Profile(directory)
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
    control floor. A bar ending between the grey ticks is a cell saying
    nothing, which is the judgement the text makes over and over and which a
    table leaves the reader to make for themselves.

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
        "\\caption{Marginal cost of one additional pass on the RTX~5070, from the "
        "pass-count sweeps. Host figures come from the timestamp-free collection "
        "and device figures from the GPU-timed one. The gap between "
        "\\texttt{B-auto} and \\texttt{B-hazard} is what one redundant barrier "
        "costs; the gap to \\texttt{W-wgpu} is end-to-end.}\n"
        "\\label{fig:marginal}\n\\end{figure}\n"
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

    # The prose names devices; the tables are generated. If a machine's raw
    # data is not on this disk the tables lose its rows silently while the text
    # keeps citing them, which is the one failure mode of keeping measurements
    # out of git.
    present = {
        DEVICE_SHORT.get(name, name)
        for collection in collections
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
        print(
            "WARNING: main.tex cites devices with no data under data/raw:\n  "
            + "\n  ".join(missing)
            + "\n  Their table rows will be absent while the text still refers "
            "to them.\n  Copy those collections back before building the PDF.",
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
            "workload. The bar is the median, the two black ticks its 95\\% "
            "bootstrap interval, and the wider grey ticks plus and minus that "
            "cell's control floor. A bar ending between the grey ticks is a cell "
            "that cannot answer, whatever its own interval looks like. Note the "
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
            "because placement has nothing to remove there.",
            "fig:scope",
        ),
        "fig-marginal.tex": marginal_figure(collections),
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

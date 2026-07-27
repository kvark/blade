#!/usr/bin/env python3
"""Summarize Blade synchronization benchmark CSV files.

The script uses only the Python standard library. Samples from one benchmark
process are not independent experimental repetitions: they share clocks,
thermal state, and run order. Each collector ``repetition`` is an outer
randomized matrix block containing one separately launched process per
configuration. Comparisons pair those launches by outer block and use a
hierarchical bootstrap that resamples outer blocks first and samples within
each selected process.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Sequence


METRICS = ("start_ns", "record_ns", "submit_ns", "wait_ns", "gpu_ns")
IDENTITY_METADATA_FIELDS = (
    "implementation",
    "backend",
    "device_name",
    "driver_name",
    "driver_info",
    "software_emulated",
    "validation",
    "gpu_timing",
)
CONFIG_FIELDS = (
    "workload",
    "policy",
    "passes",
    "elements",
    "rounds",
    "width",
    "height",
)
EXPECTED_SCHEMA = "blade-sync-bench-v1"


@dataclass(frozen=True)
class Run:
    path: Path
    metadata: dict[str, str]
    rows: tuple[dict[str, str], ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="raw CSV file or directory containing CSV files",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new or empty directory for summary.csv and comparisons.csv",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="bootstrap resamples per interval (default: 10000)",
    )
    return parser.parse_args()


def find_csv_files(inputs: Iterable[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_dir():
            paths.update(
                path
                for path in input_path.rglob("*.csv")
                if path.is_file() and has_expected_schema(path)
            )
        elif input_path.is_file():
            paths.add(input_path)
        else:
            raise ValueError(f"input does not exist: {input_path}")
    return sorted(paths)


def has_expected_schema(path: Path) -> bool:
    with path.open(encoding="utf-8", newline="") as source:
        for line in source:
            if line.startswith("#"):
                fields = next(csv.reader([line[1:].strip()]))
                if len(fields) >= 2 and fields[0].strip() == "schema":
                    return fields[1].strip() == EXPECTED_SCHEMA
            elif line.strip():
                return False
    return False


def read_run(path: Path) -> Run:
    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    with path.open(encoding="utf-8", newline="") as source:
        for line in source:
            if line.startswith("#"):
                fields = next(csv.reader([line[1:].strip()]))
                if len(fields) >= 2:
                    metadata[fields[0].strip()] = ",".join(fields[1:]).strip()
            elif line.strip():
                data_lines.append(line)

    if metadata.get("schema") != EXPECTED_SCHEMA:
        raise ValueError(
            f"{path}: schema is {metadata.get('schema')!r}, "
            f"expected {EXPECTED_SCHEMA!r}"
        )
    rows = tuple(csv.DictReader(data_lines))
    if not rows:
        raise ValueError(f"{path}: no measurement rows")
    missing = set(CONFIG_FIELDS + METRICS) - rows[0].keys()
    if missing:
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")
    return Run(path=path, metadata=metadata, rows=rows)


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def seeded_rng(parts: Sequence[str]) -> random.Random:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little"))


def _resampled_median(values: Sequence[float], rng: random.Random) -> float:
    """Median of one ordinary bootstrap resample."""
    size = len(values)
    if size == 0:
        raise ValueError("cannot resample an empty block")
    return statistics.median(values[rng.randrange(size)] for _ in range(size))


def paired_hierarchical_intervals(
    baseline: Mapping[Hashable, Sequence[float]],
    contender: Mapping[Hashable, Sequence[float]],
    count: int,
    seed_parts: Sequence[str],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return paired absolute and relative effects with hierarchical intervals.

    The point estimate is the median of the outer-block paired effects. Each
    bootstrap draw samples the common randomized matrix blocks with replacement,
    then independently resamples observations within each selected baseline
    and contender process. The two configurations are paired at the repetition
    level but not sample-by-sample: sample 7 in two separately launched
    processes has no meaningful pairing.

    The return value is ``((difference, low, high), (percent, low, high))``.
    Relative effects are percentages of the baseline. They are ``nan`` when a
    block has a zero baseline median.
    """
    common = sorted(set(baseline) & set(contender), key=str)
    if not common:
        raise ValueError("baseline and contender have no common repetitions")
    for block in common:
        if not baseline[block] or not contender[block]:
            raise ValueError(f"empty samples in repetition {block!r}")

    def effects(
        baseline_median: float, contender_median: float
    ) -> tuple[float, float]:
        difference = contender_median - baseline_median
        relative = (
            100.0 * difference / baseline_median
            if baseline_median != 0.0
            else float("nan")
        )
        return difference, relative

    observed = [
        effects(statistics.median(baseline[block]), statistics.median(contender[block]))
        for block in common
    ]
    difference_point = statistics.median(value[0] for value in observed)
    relative_observed = [value[1] for value in observed]
    relative_valid = all(value == value for value in relative_observed)
    relative_point = (
        statistics.median(relative_observed) if relative_valid else float("nan")
    )

    rng = seeded_rng(seed_parts)
    differences: list[float] = []
    relatives: list[float] = []
    block_count = len(common)
    for _ in range(count):
        selected = [common[rng.randrange(block_count)] for _ in range(block_count)]
        draw = [
            effects(
                _resampled_median(baseline[block], rng),
                _resampled_median(contender[block], rng),
            )
            for block in selected
        ]
        differences.append(statistics.median(value[0] for value in draw))
        if relative_valid and all(value[1] == value[1] for value in draw):
            relatives.append(statistics.median(value[1] for value in draw))

    difference_interval = (
        difference_point,
        percentile(differences, 0.025),
        percentile(differences, 0.975),
    )
    relative_interval = (
        (
            relative_point,
            percentile(relatives, 0.025),
            percentile(relatives, 0.975),
        )
        if len(relatives) == count
        else (float("nan"), float("nan"), float("nan"))
    )
    return difference_interval, relative_interval


def hierarchical_median_interval(
    blocks: Mapping[Hashable, Sequence[float]],
    count: int,
    seed_parts: Sequence[str],
) -> tuple[float, float, float]:
    """Median of process medians and its hierarchical bootstrap interval."""
    present = sorted((key for key, values in blocks.items() if values), key=str)
    if not present:
        raise ValueError("cannot summarize an empty group")
    point = statistics.median(statistics.median(blocks[key]) for key in present)
    rng = seeded_rng(seed_parts)
    block_count = len(present)
    medians = []
    for _ in range(count):
        selected = [present[rng.randrange(block_count)] for _ in range(block_count)]
        medians.append(
            statistics.median(_resampled_median(blocks[key], rng) for key in selected)
        )
    return point, percentile(medians, 0.025), percentile(medians, 0.975)


def group_key(run: Run, row: dict[str, str]) -> tuple[str, ...]:
    return (
        *(run.metadata.get(field, "") for field in IDENTITY_METADATA_FIELDS),
        *(row[field] for field in CONFIG_FIELDS),
    )


def write_summary(
    path: Path,
    groups: dict[tuple[str, ...], dict[str, list[float]]],
    blocked_groups: Mapping[
        tuple[str, ...], Mapping[Hashable, Mapping[str, Sequence[float]]]
    ],
    bootstrap_samples: int,
) -> None:
    identity_fields = (*IDENTITY_METADATA_FIELDS, *CONFIG_FIELDS)
    metric_fields = tuple(
        field
        for metric in METRICS
        for field in (
            f"{metric}_median",
            f"{metric}_q1",
            f"{metric}_q3",
            f"{metric}_ci_low",
            f"{metric}_ci_high",
        )
    )
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=(
                *identity_fields,
                "process_count",
                "sample_count",
                *metric_fields,
            ),
        )
        writer.writeheader()
        for key in sorted(groups):
            metrics = groups[key]
            output = dict(zip(identity_fields, key, strict=True))
            output["process_count"] = len(blocked_groups[key])
            output["sample_count"] = len(metrics[METRICS[0]])
            for metric in METRICS:
                values = metrics[metric]
                point, low, high = hierarchical_median_interval(
                    {
                        block: block_metrics[metric]
                        for block, block_metrics in blocked_groups[key].items()
                    },
                    bootstrap_samples,
                    (*key, metric),
                )
                output[f"{metric}_median"] = point
                output[f"{metric}_q1"] = percentile(values, 0.25)
                output[f"{metric}_q3"] = percentile(values, 0.75)
                output[f"{metric}_ci_low"] = low
                output[f"{metric}_ci_high"] = high
            writer.writerow(output)


def write_comparisons(
    path: Path,
    groups: Mapping[
        tuple[str, ...], Mapping[Hashable, Mapping[str, Sequence[float]]]
    ],
    bootstrap_samples: int,
) -> None:
    implementation_index = IDENTITY_METADATA_FIELDS.index("implementation")
    policy_index = len(IDENTITY_METADATA_FIELDS) + CONFIG_FIELDS.index("policy")
    by_experiment: dict[
        tuple[str, ...],
        dict[
            tuple[str, str],
            Mapping[Hashable, Mapping[str, Sequence[float]]],
        ],
    ] = defaultdict(dict)
    for key, metrics in groups.items():
        implementation = key[implementation_index]
        policy = key[policy_index]
        experiment = tuple(
            value
            for index, value in enumerate(key)
            if index not in (implementation_index, policy_index)
        )
        by_experiment[experiment][implementation, policy] = metrics

    experiment_fields = (
        *(
            field
            for field in IDENTITY_METADATA_FIELDS
            if field != "implementation"
        ),
        *(field for field in CONFIG_FIELDS if field != "policy"),
    )
    fieldnames = (
        *experiment_fields,
        "baseline_implementation",
        "baseline_policy",
        "contender_implementation",
        "contender_policy",
        "metric",
        "paired_processes",
        "baseline_samples",
        "contender_samples",
        "baseline_median",
        "contender_median",
        "difference",
        "difference_percent",
        "difference_ci_low",
        "difference_ci_high",
        "difference_percent_ci_low",
        "difference_percent_ci_high",
    )
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in sorted(by_experiment):
            configurations = by_experiment[experiment]
            comparison_pairs = (
                # Scope at fixed placement.
                (("blade", "automatic"), ("blade", "automatic-scoped")),
                (("blade", "hazard-only"), ("blade", "hazard-only-scoped")),
                (("blade", "explicit-all"), ("blade", "explicit-all-scoped")),
                # Placement at fixed scope.
                (("blade", "automatic"), ("blade", "hazard-only")),
                (("blade", "automatic-scoped"), ("blade", "hazard-only-scoped")),
                # Instrumentation controls, one per scope.
                (("blade", "automatic"), ("blade", "explicit-all")),
                (("blade", "automatic-scoped"), ("blade", "explicit-all-scoped")),
                # End-to-end.
                (("blade", "automatic"), ("wgpu", "tracked")),
            )
            for baseline_key, contender_key in comparison_pairs:
                baseline = configurations.get(baseline_key)
                contender = configurations.get(contender_key)
                if baseline is None or contender is None:
                    continue
                identity = dict(zip(experiment_fields, experiment, strict=True))
                for metric in METRICS:
                    common = sorted(set(baseline) & set(contender), key=str)
                    if not common:
                        continue
                    baseline_blocks = {
                        block: baseline[block][metric] for block in common
                    }
                    contender_blocks = {
                        block: contender[block][metric] for block in common
                    }
                    baseline_values = [
                        value for block in common for value in baseline_blocks[block]
                    ]
                    contender_values = [
                        value for block in common for value in contender_blocks[block]
                    ]
                    baseline_median = statistics.median(
                        statistics.median(baseline_blocks[block]) for block in common
                    )
                    contender_median = statistics.median(
                        statistics.median(contender_blocks[block]) for block in common
                    )
                    difference_result, relative_result = (
                        paired_hierarchical_intervals(
                            baseline_blocks,
                            contender_blocks,
                            bootstrap_samples,
                            (*experiment, *baseline_key, *contender_key, metric),
                        )
                    )
                    difference, low, high = difference_result
                    difference_percent, percent_low, percent_high = relative_result
                    writer.writerow(
                        {
                            **identity,
                            "baseline_implementation": baseline_key[0],
                            "baseline_policy": baseline_key[1],
                            "contender_implementation": contender_key[0],
                            "contender_policy": contender_key[1],
                            "metric": metric,
                            "paired_processes": len(common),
                            "baseline_samples": len(baseline_values),
                            "contender_samples": len(contender_values),
                            "baseline_median": baseline_median,
                            "contender_median": contender_median,
                            "difference": difference,
                            "difference_percent": difference_percent,
                            "difference_ci_low": low,
                            "difference_ci_high": high,
                            "difference_percent_ci_low": percent_low,
                            "difference_percent_ci_high": percent_high,
                        }
                    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if arguments.output.exists() and any(arguments.output.iterdir()):
        raise ValueError(f"output directory is not empty: {arguments.output}")
    arguments.output.mkdir(parents=True, exist_ok=True)

    files = find_csv_files(arguments.inputs)
    if not files:
        raise ValueError("no CSV files found")
    runs = [read_run(path) for path in files]

    groups: dict[tuple[str, ...], dict[str, list[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRICS}
    )
    blocked_groups: dict[
        tuple[str, ...], dict[tuple[str, str], dict[str, list[float]]]
    ] = defaultdict(
        lambda: defaultdict(lambda: {metric: [] for metric in METRICS})
    )
    for run in runs:
        block = (
            run.metadata.get("collection_id", run.path.parent.name),
            run.metadata.get("repetition", run.path.name),
        )
        for row in run.rows:
            key = group_key(run, row)
            metrics = groups[key]
            block_metrics = blocked_groups[key][block]
            for metric in METRICS:
                value = float(row[metric])
                metrics[metric].append(value)
                block_metrics[metric].append(value)

    write_summary(
        arguments.output / "summary.csv",
        groups,
        blocked_groups,
        arguments.bootstrap_samples,
    )
    write_comparisons(
        arguments.output / "comparisons.csv",
        blocked_groups,
        arguments.bootstrap_samples,
    )
    print(
        f"Read {len(files)} files and wrote {len(groups)} grouped configurations "
        f"to {arguments.output}"
    )


if __name__ == "__main__":
    main()

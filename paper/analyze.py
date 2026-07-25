#!/usr/bin/env python3
"""Summarize Blade synchronization benchmark CSV files.

The script uses only the Python standard library. It combines repetitions with
identical machine/configuration metadata, reports robust summaries, and
computes unpaired bootstrap differences from the `automatic` policy.
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
from typing import Iterable, Sequence


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


def bootstrap_median_interval(
    values: Sequence[float], count: int, seed_parts: Sequence[str]
) -> tuple[float, float]:
    rng = seeded_rng(seed_parts)
    size = len(values)
    medians = [
        statistics.median(values[rng.randrange(size)] for _ in range(size))
        for _ in range(count)
    ]
    return percentile(medians, 0.025), percentile(medians, 0.975)


def bootstrap_difference_interval(
    automatic: Sequence[float],
    contender: Sequence[float],
    count: int,
    seed_parts: Sequence[str],
) -> tuple[float, float]:
    rng = seeded_rng(seed_parts)
    automatic_size = len(automatic)
    contender_size = len(contender)
    differences = []
    for _ in range(count):
        automatic_median = statistics.median(
            automatic[rng.randrange(automatic_size)] for _ in range(automatic_size)
        )
        contender_median = statistics.median(
            contender[rng.randrange(contender_size)] for _ in range(contender_size)
        )
        differences.append(contender_median - automatic_median)
    return percentile(differences, 0.025), percentile(differences, 0.975)


def group_key(run: Run, row: dict[str, str]) -> tuple[str, ...]:
    return (
        *(run.metadata.get(field, "") for field in IDENTITY_METADATA_FIELDS),
        *(row[field] for field in CONFIG_FIELDS),
    )


def write_summary(
    path: Path,
    groups: dict[tuple[str, ...], dict[str, list[float]]],
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
            fieldnames=(*identity_fields, "sample_count", *metric_fields),
        )
        writer.writeheader()
        for key in sorted(groups):
            metrics = groups[key]
            output = dict(zip(identity_fields, key, strict=True))
            output["sample_count"] = len(metrics[METRICS[0]])
            for metric in METRICS:
                values = metrics[metric]
                low, high = bootstrap_median_interval(
                    values, bootstrap_samples, (*key, metric)
                )
                output[f"{metric}_median"] = statistics.median(values)
                output[f"{metric}_q1"] = percentile(values, 0.25)
                output[f"{metric}_q3"] = percentile(values, 0.75)
                output[f"{metric}_ci_low"] = low
                output[f"{metric}_ci_high"] = high
            writer.writerow(output)


def write_comparisons(
    path: Path,
    groups: dict[tuple[str, ...], dict[str, list[float]]],
    bootstrap_samples: int,
) -> None:
    implementation_index = IDENTITY_METADATA_FIELDS.index("implementation")
    policy_index = len(IDENTITY_METADATA_FIELDS) + CONFIG_FIELDS.index("policy")
    by_experiment: dict[
        tuple[str, ...], dict[tuple[str, str], dict[str, list[float]]]
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
        "baseline_samples",
        "contender_samples",
        "baseline_median",
        "contender_median",
        "difference",
        "difference_percent",
        "difference_ci_low",
        "difference_ci_high",
    )
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for experiment in sorted(by_experiment):
            configurations = by_experiment[experiment]
            comparison_pairs = (
                (("blade", "automatic"), ("blade", "hazard-only")),
                (("blade", "automatic"), ("blade", "explicit-all")),
                (("blade", "automatic"), ("wgpu", "tracked")),
            )
            for baseline_key, contender_key in comparison_pairs:
                baseline = configurations.get(baseline_key)
                contender = configurations.get(contender_key)
                if baseline is None or contender is None:
                    continue
                identity = dict(zip(experiment_fields, experiment, strict=True))
                for metric in METRICS:
                    baseline_values = baseline[metric]
                    contender_values = contender[metric]
                    baseline_median = statistics.median(baseline_values)
                    contender_median = statistics.median(contender_values)
                    difference = contender_median - baseline_median
                    difference_percent = (
                        100.0 * difference / baseline_median
                        if baseline_median != 0
                        else float("nan")
                    )
                    low, high = bootstrap_difference_interval(
                        baseline_values,
                        contender_values,
                        bootstrap_samples,
                        (*experiment, *baseline_key, *contender_key, metric),
                    )
                    writer.writerow(
                        {
                            **identity,
                            "baseline_implementation": baseline_key[0],
                            "baseline_policy": baseline_key[1],
                            "contender_implementation": contender_key[0],
                            "contender_policy": contender_key[1],
                            "metric": metric,
                            "baseline_samples": len(baseline_values),
                            "contender_samples": len(contender_values),
                            "baseline_median": baseline_median,
                            "contender_median": contender_median,
                            "difference": difference,
                            "difference_percent": difference_percent,
                            "difference_ci_low": low,
                            "difference_ci_high": high,
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
    for run in runs:
        for row in run.rows:
            metrics = groups[group_key(run, row)]
            for metric in METRICS:
                metrics[metric].append(float(row[metric]))

    write_summary(
        arguments.output / "summary.csv", groups, arguments.bootstrap_samples
    )
    write_comparisons(
        arguments.output / "comparisons.csv", groups, arguments.bootstrap_samples
    )
    print(
        f"Read {len(files)} files and wrote {len(groups)} grouped configurations "
        f"to {arguments.output}"
    )


if __name__ == "__main__":
    main()

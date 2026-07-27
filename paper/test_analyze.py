#!/usr/bin/env python3
"""Regression tests for the process-blocked benchmark estimator."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("analyze", HERE / "analyze.py")
assert SPEC is not None and SPEC.loader is not None
analyze = importlib.util.module_from_spec(SPEC)
sys.modules["analyze"] = analyze
SPEC.loader.exec_module(analyze)


class HierarchicalBootstrapTests(unittest.TestCase):
    def test_relative_point_is_median_of_paired_process_effects(self) -> None:
        baseline = {
            "r1": [99.0, 100.0, 101.0],
            "r2": [199.0, 200.0, 201.0],
            "r3": [399.0, 400.0, 401.0],
        }
        contender = {
            "r1": [89.0, 90.0, 91.0],
            "r2": [179.0, 180.0, 181.0],
            "r3": [359.0, 360.0, 361.0],
        }

        _, relative = analyze.paired_hierarchical_intervals(
            baseline, contender, 1_000, ("paired-ten-percent",)
        )

        self.assertAlmostEqual(relative[0], -10.0)
        self.assertLess(relative[1], 0.0)
        self.assertLess(relative[2], 0.0)

    def test_process_pairing_is_not_sample_pooling(self) -> None:
        # Two fast baseline processes and one slow one make a pooled ratio a
        # different estimand. The process-level effects are +10, -10, +10, so
        # their median must be +10 regardless of the absolute clock regimes.
        baseline = {
            "r1": [100.0] * 5,
            "r2": [1_000.0] * 5,
            "r3": [100.0] * 5,
        }
        contender = {
            "r1": [110.0] * 5,
            "r2": [900.0] * 5,
            "r3": [110.0] * 5,
        }

        _, relative = analyze.paired_hierarchical_intervals(
            baseline, contender, 500, ("paired-regimes",)
        )

        self.assertAlmostEqual(relative[0], 10.0)

    def test_seed_makes_interval_reproducible(self) -> None:
        baseline = {"r1": [1.0, 2.0, 3.0], "r2": [2.0, 3.0, 4.0]}
        contender = {"r1": [2.0, 3.0, 4.0], "r2": [3.0, 4.0, 5.0]}

        first = analyze.paired_hierarchical_intervals(
            baseline, contender, 500, ("deterministic",)
        )
        second = analyze.paired_hierarchical_intervals(
            baseline, contender, 500, ("deterministic",)
        )

        self.assertEqual(first, second)

    def test_zero_baseline_has_no_relative_interval(self) -> None:
        absolute, relative = analyze.paired_hierarchical_intervals(
            {"r1": [0.0, 0.0, 0.0]},
            {"r1": [1.0, 1.0, 1.0]},
            100,
            ("zero-baseline",),
        )

        self.assertEqual(absolute, (1.0, 1.0, 1.0))
        self.assertTrue(all(math.isnan(value) for value in relative))


if __name__ == "__main__":
    unittest.main()

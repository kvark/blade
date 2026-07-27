import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("build-tables.py")
SPEC = importlib.util.spec_from_file_location("build_tables", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_tables = importlib.util.module_from_spec(SPEC)
sys.modules["build_tables"] = build_tables
SPEC.loader.exec_module(build_tables)


class ProcessLevelSummaryTests(unittest.TestCase):
    def test_absolute_point_is_median_of_process_medians(self) -> None:
        class FakeCollection:
            def blocked_values(self, *_args):
                return {
                    "r1": [900.0, 1_000.0, 1_100.0],
                    "r2": [1_900.0, 2_000.0, 2_100.0],
                    "r3": [9_900.0, 10_000.0, 10_100.0],
                }

        self.assertEqual(
            build_tables.process_median_us(
                FakeCollection(),
                "blade",
                "automatic",
                "compute-independent",
                16,
                "gpu_ns",
            ),
            2.0,
        )

    def test_dispersion_is_computed_within_each_process(self) -> None:
        collection = object.__new__(build_tables.Collection)
        key = ("blade", "automatic", "compute-independent", 16)
        collection.block_samples = {
            key: {
                "r1": {"host_ns": [100.0, 100.0, 100.0, 100.0]},
                "r2": {"host_ns": [1_000.0, 1_000.0, 1_000.0, 1_000.0]},
            }
        }

        # Pooling the two clock regimes would report a large spread. Each
        # separately launched process is internally constant, so the intended
        # within-process diagnostic is zero.
        self.assertEqual(collection.dispersion("host_ns"), 0.0)


class StabilityFloorTests(unittest.TestCase):
    def test_interval_must_clear_the_same_side_of_the_floor(self) -> None:
        self.assertTrue(
            build_tables.clears_stability_floor((-5.0, -6.0, -3.0), 2.0)
        )
        self.assertTrue(
            build_tables.clears_stability_floor((5.0, 3.0, 6.0), 2.0)
        )

    def test_point_beyond_floor_is_not_enough(self) -> None:
        self.assertFalse(
            build_tables.clears_stability_floor((-5.0, -8.0, -0.5), 2.0)
        )


class ProfileMetadataTests(unittest.TestCase):
    def test_adapter_mismatch_is_visible_by_implementation(self) -> None:
        profile = object.__new__(build_tables.Profile)
        profile.manifest = {
            "devices": {
                "blade/compute-independent": (
                    "AMD Ryzen 5 9600X 6-Core Processor (RADV RAPHAEL_MENDOCINO)"
                ),
                "wgpu/compute-independent": "AMD Radeon RX 7900 XT (RADV NAVI31)",
            }
        }
        self.assertEqual(
            profile.devices_by_implementation,
            {
                "blade": {"Raphael iGPU"},
                "wgpu": {"RX 7900 XT"},
            },
        )


if __name__ == "__main__":
    unittest.main()

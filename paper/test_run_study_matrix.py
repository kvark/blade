import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run-study-matrix.py")
SPEC = importlib.util.spec_from_file_location("run_study_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
run_study_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_study_matrix)


class ExtractBenchmarkCsvTests(unittest.TestCase):
    def test_validation_diagnostics_are_kept_out_of_csv(self) -> None:
        source = """\
Validation Warning: [ VALIDATION-SETTINGS ]
vkCreateInstance(): diagnostic with, commas

# schema,blade-sync-bench-v1
# implementation,wgpu
sample,workload,policy
0,graphics-independent,tracked
an interleaved diagnostic
1,graphics-independent,tracked
# validation_hash,fnv1a64-standard:0123456789abcdef
"""
        self.assertEqual(
            run_study_matrix.extract_benchmark_csv(source),
            """\
# schema,blade-sync-bench-v1
# implementation,wgpu
sample,workload,policy
0,graphics-independent,tracked
1,graphics-independent,tracked
# validation_hash,fnv1a64-standard:0123456789abcdef
""",
        )


class SystemInfoRedactionTests(unittest.TestCase):
    def test_macos_private_identifiers_are_redacted(self) -> None:
        source = """\
      Model Identifier: Mac15,12
      Serial Number (system): SERIAL
      Hardware UUID: HARDWARE-UUID
      Provisioning UDID: PROVISIONING-UDID
      Activation Lock Status: Enabled
      Chip: Apple M3
"""
        redacted = run_study_matrix.redact_macos_system_identifiers(source)
        self.assertNotIn("SERIAL", redacted)
        self.assertNotIn("HARDWARE-UUID", redacted)
        self.assertNotIn("PROVISIONING-UDID", redacted)
        self.assertNotIn("Enabled", redacted)
        self.assertIn("Model Identifier: Mac15,12", redacted)
        self.assertIn("Chip: Apple M3", redacted)
        self.assertEqual(redacted.count("[redacted from public artifact]"), 4)


class BevyFamilyTests(unittest.TestCase):
    def test_bevy_workload_name(self) -> None:
        self.assertEqual(run_study_matrix.BEVY_WORKLOAD, "bevy-headless")

    def test_synthetic_workloads_still_agree_on_hash(self) -> None:
        for workload in run_study_matrix.SHARED_WORKLOADS:
            self.assertTrue(
                run_study_matrix.participates_in_hash_agreement(workload)
            )
        for workload in run_study_matrix.BLADE_ONLY_WORKLOADS:
            self.assertTrue(
                run_study_matrix.participates_in_hash_agreement(workload)
            )

    def test_bevy_family_is_excluded_from_shader_hash_agreement(self) -> None:
        self.assertFalse(
            run_study_matrix.participates_in_hash_agreement(
                run_study_matrix.BEVY_WORKLOAD
            )
        )

    def test_bevy_csv_rows_are_retained(self) -> None:
        source = """\
# schema,blade-sync-bench-v1
# implementation,blade
# gpu_timing_method,wait-to-idle
sample,workload,policy,passes,elements,rounds,width,height,start_ns,record_ns,submit_ns,wait_ns,gpu_ns,gpu_pass_count
0,bevy-headless,automatic,6,1,1,1280,720,0,100,50,200,200,6
# validation_hash,fnv1a64-standard:0123456789abcdef
"""
        csv = run_study_matrix.extract_benchmark_csv(source)
        self.assertIn("bevy-headless", csv)
        self.assertIn("gpu_pass_count", csv)
        self.assertIn("wait-to-idle", csv)


if __name__ == "__main__":
    unittest.main()

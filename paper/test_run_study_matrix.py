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


if __name__ == "__main__":
    unittest.main()

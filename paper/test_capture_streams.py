#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "capture_streams", HERE / "capture-streams.py"
)
assert SPEC is not None and SPEC.loader is not None
CAPTURE_STREAMS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CAPTURE_STREAMS)


class CaptureExtractionTests(unittest.TestCase):
    def extract(self, chunks: str) -> list[dict]:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.xml"
            path.write_text(f"<root>{chunks}</root>", encoding="utf-8")
            return CAPTURE_STREAMS.extract_barriers(
                path, "wgpu", "compute-chain", "tracked"
            )

    def test_placement_uses_submission_not_recording_order(self) -> None:
        rows = self.extract(
            """
            <chunk name="vkCmdDispatch">
              <ResourceId name="commandBuffer">work-a</ResourceId>
            </chunk>
            <chunk name="vkCmdDispatch">
              <ResourceId name="commandBuffer">work-b</ResourceId>
            </chunk>
            <chunk name="vkCmdPipelineBarrier">
              <ResourceId name="commandBuffer">transition</ResourceId>
              <uint name="memoryBarrierCount">1</uint>
              <uint name="bufferMemoryBarrierCount">0</uint>
              <uint name="imageMemoryBarrierCount">0</uint>
              <enum name="srcStageMask">65536</enum>
              <enum name="destStageMask">65536</enum>
            </chunk>
            <chunk name="vkQueueSubmit">
              <array name="pCommandBuffers">
                <ResourceId>work-a</ResourceId>
                <ResourceId>transition</ResourceId>
                <ResourceId>work-b</ResourceId>
              </array>
            </chunk>
            """
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["after_work"], 1)
        self.assertEqual(rows[0]["index"], 0)

    def test_unsubmitted_barrier_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "recorded 1 barriers"):
            self.extract(
                """
                <chunk name="vkCmdPipelineBarrier">
                  <ResourceId name="commandBuffer">not-submitted</ResourceId>
                  <uint name="memoryBarrierCount">1</uint>
                </chunk>
                <chunk name="vkQueueSubmit">
                  <array name="pCommandBuffers">
                    <ResourceId>something-else</ResourceId>
                  </array>
                </chunk>
                """
            )


class BenchmarkMetadataTests(unittest.TestCase):
    def test_csv_quoting_is_preserved(self) -> None:
        metadata = CAPTURE_STREAMS.parse_benchmark_metadata(
            '# device_name,"GPU, Inc."\n'
            "# implementation,blade\n"
            "sample,workload\n"
        )
        self.assertEqual(metadata["device_name"], "GPU, Inc.")
        self.assertEqual(metadata["implementation"], "blade")


if __name__ == "__main__":
    unittest.main()

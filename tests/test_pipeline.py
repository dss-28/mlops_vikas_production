# tests/test_pipeline.py
import unittest

class TestPipelineEntrypoint(unittest.TestCase):

    def test_pipeline_entrypoint_exists(self):
        import pipeline
        self.assertTrue(
            hasattr(pipeline, "main"),
            "pipeline.main() is missing"
        )

if __name__ == "__main__":
    unittest.main()

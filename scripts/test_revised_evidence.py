"""Regression test: uploaded Colab records match the submitted numerical claims."""
import unittest
from verify_revised_evidence import verify

class TestRevisedEvidence(unittest.TestCase):
    def test_uploaded_records(self):
        report=verify()
        self.assertEqual(report['revised']['raw_json_valid'], 32)
        self.assertEqual(report['revised']['raw_source_exact'], 4)
        self.assertEqual(report['revised']['fallback'], 28)
        self.assertEqual(report['natural_language_retrieval']['correct'], 0)

if __name__ == '__main__':
    unittest.main()

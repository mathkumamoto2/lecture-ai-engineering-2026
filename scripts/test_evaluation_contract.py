"""Run: python -m unittest discover -s scripts -p 'test_*.py' -v"""
import csv
import json
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_contract import (
    accept_grounded_generation, lookup_exact_quoted, parse_model_json,
    source_grounded_answer, validate_output,
)

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / 'lectures' / '02' / 'evidence'


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.glossary = [
            {'id':'api','term':'API','aliases':['Application Programming Interface'], 'definition':'APIの定義'},
            {'id':'api-key','term':'APIキー','aliases':['API key'], 'definition':'APIキーの定義'},
        ]

    def test_exact_schema(self):
        self.assertTrue(validate_output({'answer':'ok','needs_review':False}))
        for value in [
            {'answer':'ok','needs_review':'false'},
            {'answer':'ok'},
            {'key':'ok','needs_review':False},
            {'answer':None,'needs_review':False},
        ]:
            with self.subTest(value=value):
                self.assertFalse(validate_output(value))

    def test_parse_invalid(self):
        self.assertIsNone(parse_model_json('{oops'))
        self.assertIsNone(parse_model_json('{"key":"API","needs_review":"false"}'))

    def test_exact_term_not_substring(self):
        a, name = source_grounded_answer('「API」とは何ですか', self.glossary)
        self.assertEqual(name, 'api')
        self.assertEqual(a['answer'],'APIの定義')
        _, name = source_grounded_answer('「APIキー」とは何ですか', self.glossary)
        self.assertEqual(name, 'api-key')

    def test_abstain_on_unknown_and_unquoted(self):
        a, name = source_grounded_answer('「DPO」とは何ですか', self.glossary)
        self.assertIsNone(name)
        self.assertTrue(a['needs_review'])
        self.assertIsNone(lookup_exact_quoted('秘密の認証文字列は何ですか', self.glossary))

    def test_ambiguous_alias_abstains(self):
        overlapping=self.glossary + [{'id':'extra','term':'API','definition':'other'}]
        self.assertIsNone(lookup_exact_quoted('「API」とは？', overlapping))

    def test_model_output_never_overrides_source_without_check(self):
        source={'answer':'APIの定義','needs_review':False}
        out,fallback=accept_grounded_generation('{"answer":"創作","needs_review":false}',source)
        self.assertTrue(fallback)
        self.assertEqual(out,source)
        out,fallback=accept_grounded_generation(json.dumps(source,ensure_ascii=False),source)
        self.assertFalse(fallback)
        self.assertEqual(out,source)

    def test_saved_replay_is_structurally_valid(self):
        with (EVID/'deterministic_source_replay.csv').open(encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),32)
        for row in rows:
            self.assertTrue(validate_output(json.loads(row['output_json'])))
        self.assertEqual(sum(json.loads(r['output_json'])['needs_review'] for r in rows),8)

    def test_saved_metrics_are_consistent(self):
        src=json.loads((EVID/'final_audit_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(src['original']['before']['format_valid'],32)
        self.assertEqual(src['original']['after']['format_valid'],29)
        self.assertEqual(src['rag_llm_raw_valid_json'],0)
        self.assertEqual(src['retrieval_hits_correct'],32)

if __name__ == '__main__':
    unittest.main()

"""Recompute observation counts from immutable recorded inference outputs.

Usage: python scripts/audit_saved_results.py
Runs without GPU, network, external libraries, or secrets.
"""
from __future__ import annotations
import csv
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVID=ROOT/'lectures'/'02'/'evidence'

def main() -> None:
    source={}
    for label in ('before','after','rag'):
        data=json.loads((EVID/f'eval_{label}.json').read_text(encoding='utf-8'))
        source[label]={
            'questions':len(data['rows']),
            'format_valid':sum(row['format_ok'] for row in data['rows']),
            'decision_correct_original_rule':sum(row['decision_ok'] for row in data['rows']),
            'original_both_rate_is_same_metric':all(row['both_ok']==row['decision_ok'] for row in data['rows'])
        }
    with (EVID/'rag_diagnostics.csv').open(encoding='utf-8-sig',newline='') as f:
        rag=list(csv.DictReader(f))
    with (EVID/'semantic_first_pass.csv').open(encoding='utf-8-sig',newline='') as f:
        reviewed=list(csv.DictReader(f))
    source['retrieval']={
        'n':len(rag),
        'expected_id_matches':sum(r['retrieval_correct']=='True' for r in rag),
        'json_valid_in_original_llm_output':sum(r['rag_json_valid']=='True' for r in rag),
    }
    known=[r for r in reviewed if r['category']!='unknown']
    source['preliminary_ai_factual_review']={
        'n':len(known),
        'before':dict(Counter(r['before_label'] for r in known)),
        'after':dict(Counter(r['after_label'] for r in known)),
        'NOT_HUMAN_VERIFIED':True,
    }
    assert source['before']['format_valid']==32
    assert source['after']['format_valid']==29
    assert source['rag']['format_valid']==0
    assert source['retrieval']['expected_id_matches']==32
    assert source['retrieval']['json_valid_in_original_llm_output']==0
    assert source['preliminary_ai_factual_review']['n']==24
    print(json.dumps(source,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()

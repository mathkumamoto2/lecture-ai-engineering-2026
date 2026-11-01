"""Validate immutable experiment evidence from an uploaded Colab run.

This script recalculates metrics from individual outputs. It requires no GPU,
network, model access, or third-party packages.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from evaluation_contract import validate_output, parse_model_json

ROOT=Path(__file__).resolve().parents[1]
EVID=ROOT/'lectures'/'02'/'evidence'


def load(filename: str):
    return json.loads((EVID/filename).read_text(encoding='utf-8'))


def verify():
    tests=load('test_cases.json')
    before=load('eval_before.json')
    after=load('eval_after.json')
    oldrag=load('eval_rag.json')
    new=load('revised_rag_evaluation.json')
    naturals=load('natural_questions_retrieval.json')
    main=load('advanced_result.json')
    assert len(tests)==32
    assert len({x['question'] for x in tests})==32
    assert len(before['rows'])==len(after['rows'])==len(oldrag['rows'])==32
    assert len(new['rows'])==32
    assert main['executed_training_myself'] is True
    assert main['training']['steps']==48

    # Original reported both_rate is *not* a factual correctness metric.
    originals={}
    for label,source in [('before',before),('after',after),('rag',oldrag)]:
        rows=source['rows']
        assert [r['question'] for r in rows]==[x['question'] for x in tests]
        assert all(r['both_ok']==r['decision_ok'] for r in rows)
        originals[label]={
            'json_valid':sum(bool(r['format_ok']) for r in rows),
            'decision_match':sum(bool(r['decision_ok']) for r in rows),
        }
    assert originals=={'before':{'json_valid':32,'decision_match':23},
                       'after':{'json_valid':29,'decision_match':24},
                       'rag':{'json_valid':0,'decision_match':0}}

    report=Counter()
    groups={}
    for case,row in zip(tests,new['rows']):
        assert row['question']==case['question']
        assert row['expected_id']==case['term_id']
        assert row['category']==case['category']
        assert row['retrieval_correct']==(row['retrieved_id']==case['term_id'])
        parsed=parse_model_json(row['llm_raw'])
        assert row['llm_json_valid']==(parsed is not None)
        assert row['llm_matches_source']==(parsed is not None and parsed==row['served'])
        assert row['fallback_used']==(not row['llm_matches_source'])
        expected_served=({'answer':case['reference'],'needs_review':False}
                         if case['reference'] is not None
                         else {'answer':'この用語は用語集に見つかりませんでした。確認が必要です。',
                               'needs_review':True})
        assert row['served']==expected_served
        assert validate_output(row['served'])
        assert row['served_definition_equals_reference']
        assert row['retrieval_correct']
        report['retrieval_correct']+=int(row['retrieval_correct'])
        report['raw_json_valid']+=int(row['llm_json_valid'])
        report['raw_source_exact']+=int(row['llm_matches_source'])
        report['fallback']+=int(row['fallback_used'])
        report['served_valid']+=int(row['served_json_valid'])
        report['served_grounded']+=int(row['served_definition_equals_reference'])
        g=groups.setdefault(row['category'],Counter())
        g['n']+=1
        g['raw_source_exact']+=int(row['llm_matches_source'])
        g['fallback']+=int(row['fallback_used'])
        if row['category']=='unknown':
            g['model_review_true']+=int(bool(parsed['needs_review']))
    assert dict(report)=={'retrieval_correct':32,'raw_json_valid':32,
                         'raw_source_exact':4,'fallback':28,
                         'served_valid':32,'served_grounded':32}
    assert len(naturals)==8
    assert all(row['retrieved_id'] is None for row in naturals)
    for key,stored_key in [('retrieval_correct','retrieval_id_correct'),
                           ('raw_json_valid','new_llm_json_valid'),
                           ('raw_source_exact','new_llm_source_exact'),
                           ('fallback','fallback_used'),
                           ('served_valid','served_json_valid'),
                           ('served_grounded','served_grounded_or_abstained')]:
        assert report[key]==new['summary'][stored_key]
    assert main['evaluations']['before']['format_rate']==1.0
    return {'original':originals, 'revised':dict(report),
            'groups':{k:dict(v) for k,v in sorted(groups.items())},
            'natural_language_retrieval':{'correct':0,'n':len(naturals)},
            'training':main['training']}

if __name__=='__main__':
    print(json.dumps(verify(),ensure_ascii=False,indent=2))

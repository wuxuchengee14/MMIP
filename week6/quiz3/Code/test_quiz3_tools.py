"""Meaningful offline checks; synthetic completion fixtures stay in temporary folders."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import score_report as scoring
from common import GENERATION,MODELS,ROOT,VARIANTS,digest,read,save,sha,validate_data


class DataAndMetricsTests(unittest.TestCase):
    def test_frozen_data_and_exact_preference_labels(self):
        manifest=validate_data()
        self.assertEqual(manifest['counts']['general_benchmark'],100)
        for kind in ['llm','vlm']:
            train=read(ROOT/f'Data/{kind}_train.json')
            self.assertGreaterEqual(len(train),100)
            for q in train:
                if kind=='llm':
                    self.assertEqual(scoring.llm_score(q,q['chosen'])['correct'],1)
                    self.assertEqual(scoring.llm_score(q,q['rejected'])['correct'],0)
                else:
                    actual=sum(o['color']==q['target_color'] and o['shape']==q['target_shape'] for o in q['objects'])
                    self.assertEqual(actual,q['expected_count'])
                    self.assertEqual(scoring.vlm_score(q,q['chosen'])['correct'],1)
                    self.assertEqual(scoring.vlm_score(q,q['rejected'])['correct'],0)

    def test_strict_json_and_fabrication(self):
        q={'reference':{'count':1,'missing':None},'missing_keys':['missing']}
        self.assertEqual(scoring.llm_score(q,'{"count":1,"missing":null}')['correct'],1)
        for text in ['{"count":true,"missing":null}','{"count":1,"count":1,"missing":null}',
                     '{"count":1,"missing":NaN}','```json\n{"count":1,"missing":null}\n```']:
            self.assertEqual(scoring.llm_score(q,text)['correct'],0)
        self.assertEqual(scoring.llm_score(q,'{"count":1,"missing":"invented"}')['fabricated_missing'],1)

    def test_absence_and_invalid_count_separated(self):
        q={'expected_count':0}
        self.assertEqual(scoring.vlm_score(q,'0')['correct'],1)
        self.assertEqual(scoring.vlm_score(q,'3')['false_positive_absent'],1)
        answer=scoring.vlm_score(q,'No objects are visible.')
        self.assertEqual(answer['correct'],0)
        self.assertEqual(answer['format_valid'],0)
        self.assertIsNone(answer['absolute_error'])

    def test_paired_change(self):
        r=scoring.paired_change([0,1,1,0],[1,1,0,1])
        self.assertEqual(r['delta_percentage_points'],25)
        self.assertEqual(r['wrong_to_right'],2)
        self.assertEqual(r['right_to_wrong'],1)

    def test_supplementary_content_does_not_repair_format(self):
        q={'expected_count':3}
        r=scoring.vlm_score(q,'The image contains 3 red circles.')
        self.assertEqual(r['correct'],0)
        self.assertEqual(r['semantic_correct'],1)
        self.assertEqual(scoring.semantic_count('There are no blue squares.'),0)
        self.assertIsNone(scoring.semantic_count('There are 3 circles and 2 squares.'))
        self.assertIsNone(scoring.semantic_count('Maybe three circles.'))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        shutil.copytree(ROOT/'Data',self.root/'Data')
        self.patches=[patch.object(common,'ROOT',self.root),patch.object(scoring,'ROOT',self.root)]
        for p in self.patches:p.start()
        for kind in ['llm','vlm']:
            folder=self.root/f'Output/{kind}';adapter=folder/'adapter';adapter.mkdir(parents=True)
            (adapter/'adapter_model.safetensors').write_bytes(b'UNIT TEST FIXTURE, NOT MODEL WEIGHTS')
            training={'complete':True,'smoke':False,'train_examples':200,'test_examples_used_for_training':0,
                'global_step':1,'adapter_parameter_change_l2':0.01,'training_seconds':1.,
                'data_manifest_sha256':sha(self.root/'Data/manifest.json'),
                'adapter_sha256':{'adapter_model.safetensors':sha(adapter/'adapter_model.safetensors')}}
            save(folder/'training.json',training)
            target=read(self.root/f'Data/{kind}_test.json')
            if kind=='llm':questions=[dict(q,suite='target',variant='text') for q in target]+[dict(q,suite='general',variant='text') for q in read(self.root/'Data/general_100.json')]
            else:questions=[dict(q,suite='visual',variant=v) for v in VARIANTS for q in target]
            for stage in ['before','after']:
                rows=[]
                for q in questions:
                    reference=json.dumps(q['reference'],ensure_ascii=False) if isinstance(q['reference'],dict) else q['reference']
                    rows.append({k:q[k] for k in ['id','suite','variant','prompt']}|{'image_sha256':sha(self.root/f"Data/variants/{q['variant']}/{q['id']}.png") if kind=='vlm' else None,
                        'answer':reference,'seconds':0.1,'hit_token_limit':False})
                run={'complete':True,'smoke':False,'model':{k:v for k,v in MODELS[kind].items() if k!='cache'},
                    'generation_config':GENERATION,'data_manifest_sha256':sha(self.root/'Data/manifest.json'),
                    'request_digest':digest([{k:r[k] for k in ['id','suite','variant','prompt','image_sha256']} for r in rows]),
                    'gpu':'Unit test fixture GPU','packages':{},'processor_config':{},'native_token_ids':{},
                    'thinking':False if kind=='llm' else None,'training_sha256':sha(folder/'training.json') if stage=='after' else None,'results':rows}
                save(folder/(stage+'.json'),run)

    def tearDown(self):
        for p in self.patches:p.stop()
        self.tmp.cleanup()

    def test_report_complete_only_with_all_evidence(self):
        scoring.report()
        self.assertEqual(read(self.root/'Output/completion.json')['covered_points'],40)
        self.assertTrue((self.root/'Output/report.md').exists())
        self.assertEqual(read(self.root/'Output/metrics.json')['results']['llm']['before']['target']['accuracy_percent'],100)

    def test_smoke_or_unchanged_weights_do_not_count(self):
        path=self.root/'Output/llm/training.json';original=read(path)
        for key,value in [('smoke',True),('adapter_parameter_change_l2',0),('test_examples_used_for_training',1)]:
            row=copy.deepcopy(original);row[key]=value;save(path,row)
            with self.assertRaises(ValueError):scoring.audit_runs('llm')
        save(path,original)

    def test_request_change_or_adapter_tampering_rejected(self):
        path=self.root/'Output/llm/after.json';row=read(path);row['results'][0]['prompt']='Changed prompt';save(path,row)
        with self.assertRaises(ValueError):scoring.audit_runs('llm')
        (self.root/'Output/vlm/adapter/adapter_model.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):scoring.audit_runs('vlm')

    def test_missing_general_or_visual_variant_rejected(self):
        for kind in ['llm','vlm']:
            path=self.root/f'Output/{kind}/after.json';row=read(path);row['results'].pop();save(path,row)
            with self.assertRaises(ValueError):scoring.audit_runs(kind)


if __name__=='__main__':unittest.main()

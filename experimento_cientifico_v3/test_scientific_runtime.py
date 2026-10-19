"""CPU contract tests. NOT a substitute for training models on CUDA."""
import csv
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import scientific_runtime as lab
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revisao_2026/src'))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


class ScientificContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_dataset_cross_split_leakage_is_fatal(self):
        sample=[dict(item_id='1',split='train',prompt='x'),dict(item_id='2',split='test',prompt='x'),
                dict(item_id='3',split='dev',prompt='y')]
        with self.assertRaisesRegex(ValueError,'enunciado'):
            lab.dataset_audit(sample)
        sample[1]['prompt']='z'
        self.assertEqual(lab.dataset_audit(sample),dict(train=1,test=1,dev=1))

    def test_dataset_duplicate_id_fails(self):
        with self.assertRaisesRegex(ValueError,'duplicado'):
            lab.dataset_audit([dict(item_id='1',split='train',prompt='x'),
                               dict(item_id='1',split='dev',prompt='y')])

    def test_exact_signflip_two_sided_and_minimum_n(self):
        self.assertAlmostEqual(lab.signflip([1.]*8),2/256)
        self.assertAlmostEqual(lab.signflip([0.]*8),1.0)
        with self.assertRaises(ValueError):lab.signflip([1.])

    def test_nested_seed_bootstrap_does_not_count_items_as_seeds(self):
        obs=[0.10]*8
        self.assertEqual(lab.bootstrap_ci(obs,iterations=1000),[0.10,0.10])

    def test_pairing_sft_exact_only(self):
        group={}
        for seed in range(113,121):
            for level in lab.LEVELS['sft']:
                for cond in lab.CONDITIONS['sft']:
                    group[level,seed,cond]={str(i):{'accuracy':int(cond=='explicit') if level=='C' else 1}
                                              for i in range(20)}
        self.assertEqual(lab.paired_sft(group),[-1]*8)
        group['C',113,'direct'].pop('0')
        with self.assertRaisesRegex(RuntimeError,'pareados'):
            lab.paired_sft(group)

    def test_monitor_that_flags_everything_fails(self):
        model='HuggingFaceTB/SmolLM2-135M-Instruct'
        revision='a'*40
        lab.atomic(self.root/'protocol/independent_monitor.json',dict(model_id=model,revision=revision))
        rows=[dict(label=str(i%2),flag='1',split='heldout',source='human',
                   monitor_model=model,monitor_revision=revision) for i in range(200)]
        write_csv(self.root/'human'/'independent_monitor_scored.csv',rows)
        result=lab.monitor_gate(self.root)
        self.assertFalse(result['passed'])
        self.assertEqual(result['tpr'],1)
        self.assertEqual(result['fpr'],1)
        for row in rows:row['flag']=row['label']
        write_csv(self.root/'human'/'independent_monitor_scored.csv',rows)
        self.assertTrue(lab.monitor_gate(self.root)['passed'])

    def test_two_human_ratings_cannot_be_invented(self):
        self.assertFalse(lab.human_quality_gate(self.root)['passed'])
        write_csv(self.root/'human'/'blinded_items.csv',[
            dict(annotation_id=str(i),question='q'+str(i)) for i in range(360)])
        records=[dict(annotation_id=str(i),rater=r,label=str(i%2))
                 for i in range(360) for r in ('human_a','human_b')]
        write_csv(self.root/'human'/'completed_annotations.csv',records)
        self.assertTrue(lab.human_quality_gate(self.root)['passed'])
        records[0]['label']='1' if records[0]['label']=='0' else '0'
        write_csv(self.root/'human'/'completed_annotations.csv',records)
        self.assertFalse(lab.human_quality_gate(self.root)['passed'])
        write_csv(self.root/'human'/'adjudication.csv',[dict(annotation_id='0',label='0')])
        self.assertTrue(lab.human_quality_gate(self.root)['passed'])

    def test_no_report_claim_with_missing_experiment(self):
        with patch.dict(os.environ,{'COT_SOURCE_COMMIT':'47d935f2bbfeb2638c0a734f4be91ac940df9c10'}):
            lab.atomic(self.root/'protocol'/'locked_protocol.json',lab.plan_config())
            # No data or evaluations: fail closed.
            with self.assertRaises(FileNotFoundError):lab.final_report(self.root)

    def test_frozen_protocol_cannot_be_modified(self):
        with patch.dict(os.environ,{'COT_SOURCE_COMMIT':'47d935f2bbfeb2638c0a734f4be91ac940df9c10'}):
            lab.atomic(self.root/'protocol'/'locked_protocol.json',dict(lab.plan_config(),seeds=[1]))
            with self.assertRaises(RuntimeError):lab.assert_frozen(self.root,'sft')

    def test_frozen_datasets_roundtrip_and_integrity(self):
        fake_instruments=types.ModuleType('instruments')
        fake_gpu=types.ModuleType('gpu_lab');fake_benchmark=types.ModuleType('gpu_lab.benchmark')
        def make_sft(n):
            return [dict(item_id=f'{level}_{split}_{i}',prompt=f'{level} {split} {i}',
                         split=split,level=level,gold='1',dataset_seed=20261015)
                    for level in ('A','B','C') for split in ('train','dev','test')
                    for i in range(n)]
        def make_grpo(n):
            return [dict(item_id=f'{level}_{domain}_{split}_{i}',
                         question=f'{level} {domain} {split} {i}',
                         split=split,level=level,gold='A',dataset_seed=20261015)
                    for level in ('low','medium','high')
                    for domain in ('arithmetic','program_trace','fictional_rule','inventory')
                    for split in ('train','dev','test') for i in range(n)]
        fake_instruments.make_disjoint_data=make_sft
        fake_benchmark.create_dataset=make_grpo
        with patch.dict(os.environ,{'COT_SOURCE_COMMIT':'47d935f2bbfeb2638c0a734f4be91ac940df9c10'}), \
             patch.dict(sys.modules,{'instruments':fake_instruments,
                                     'gpu_lab':fake_gpu,'gpu_lab.benchmark':fake_benchmark}):
            result=lab.make_plan(self.root)
            self.assertEqual(result['datasets']['sft']['test_per_level']['A'],220)
            self.assertEqual(result['datasets']['grpo']['test_per_level']['high'],240)
            self.assertEqual(result,lab.make_plan(self.root))
            with (self.root/'data/sft.jsonl').open('a') as f:f.write('poison\n')
            with self.assertRaisesRegex(ValueError,'Dados bloqueados'):
                lab.make_plan(self.root)

    def test_all_five_notebook_cells_parse_and_embedded_source_matches(self):
        notebooks=sorted(Path(__file__).parent.glob('*.ipynb'))
        self.assertEqual(len(notebooks),5)
        self.assertEqual(notebooks[0].name,'00_protocolo_e_datasets.ipynb')
        for path in notebooks:
            doc=json.loads(path.read_text())
            codes=[c['source'] for c in doc['cells'] if c['cell_type']=='code']
            for code in codes:compile(code,str(path),'exec')
            first=codes[0]
            self.assertIn(lab.digest(Path(lab.__file__).read_bytes()),first)
            self.assertIn('SOURCE_COMMIT',first)

    def freeze_real_study(self):
        self.env=patch.dict(os.environ,{'COT_SOURCE_COMMIT':'4b498226f0ce5a302e32dbf773a99365302149c2'})
        self.env.start()
        self.addCleanup(self.env.stop)
        result=lab.make_plan(self.root)
        models={name:'a'*40 for name in (*lab.MODEL.values(),lab.MONITOR,lab.INDEPENDENT_MONITOR)}
        lab.atomic(self.root/'protocol/model_revisions.json',models)
        lab.resolve_and_freeze_models(self.root)
        return result

    def test_real_datasets_match_registered_counts_and_resume(self):
        result=self.freeze_real_study()
        self.assertEqual(result['datasets']['sft']['test_per_level'],{'A':220,'C':220})
        self.assertEqual(result['datasets']['grpo']['test_per_level'],{'low':240,'high':240})
        self.assertEqual(len(lab.jobs(self.root,'sft')),120)
        self.assertEqual(len(lab.jobs(self.root,'grpo')),120)
        self.assertEqual(lab.make_plan(self.root),result)
        self.assertEqual(lab.study_status(self.root)['training'],{'sft':0,'grpo':0})

    def test_independent_model_is_frozen_before_any_training(self):
        self.freeze_real_study()
        config=lab.read_json(self.root/'protocol/independent_monitor.json')
        self.assertEqual(config,dict(model_id=lab.INDEPENDENT_MONITOR,revision='a'*40))
        lab.atomic(self.root/'protocol/independent_monitor.json',dict(config,revision='b'*40))
        with self.assertRaisesRegex(RuntimeError,'independente'):
            lab.resolve_and_freeze_models(self.root)

    def test_no_complete_verdict_without_training_and_humans(self):
        self.freeze_real_study()
        result=lab.final_report(self.root)
        self.assertEqual(result['verdict'],'INSUFFICIENT_EVIDENCE')
        self.assertIn('incomplete_sft',result['blocked_by'])
        self.assertIn('human_validation_incomplete',result['blocked_by'])

    def test_sft_test_waits_for_grpo_training_too(self):
        self.freeze_real_study()
        sft={j:self.root/'unused' for j in lab.jobs(self.root,'sft')}
        with patch.object(lab,'job_index',side_effect=lambda b,m:sft if m=='sft' else {}):
            with self.assertRaisesRegex(RuntimeError,'TODOS'):
                lab.evaluate_batch(self.root,'sft')

    def test_registration_rejects_retroactive_declaration(self):
        self.freeze_real_study()
        from colab_support import ScientificSession
        session=ScientificSession.__new__(ScientificSession)
        session.storage=self.root
        lab.atomic(self.root/'runs/sft/already_started/manifest.json',{})
        with self.assertRaisesRegex(RuntimeError,'retroativo'):
            session.registration('https://osf.io/example/')

    def test_independent_scores_resume_and_detect_tampering(self):
        self.freeze_real_study()
        job=('high',113,'none')
        evaluation=self.root/'runs/grpo/fixture/evaluation.jsonl'
        evaluation.parent.mkdir(parents=True)
        evaluation.write_text('fixture-input\n')
        lab.atomic(self.root/'analysis/grpo/evaluated_jobs.json',{
            '|'.join(map(str,job)):dict(path=str(evaluation),sha256=lab.digest(evaluation.read_bytes()))})
        primary={job:{'item':dict(run_id='fixture')}}
        def score(source,destination,model,revision,**kwargs):
            Path(destination).write_text(lab.stable(dict(item_id='item',run_id='fixture',
                base_only=False,monitor_model=model,monitor_revision=revision,trace_flag=True))+'\n')
        with patch.object(lab,'load_primary',return_value=primary), \
             patch('gpu_lab.independent_monitor.evaluate_monitor',side_effect=score) as detector:
            self.assertEqual(lab.independent_monitor_batch(self.root)['done'],1)
            self.assertEqual(lab.independent_monitor_batch(self.root)['done'],1)
            self.assertEqual(detector.call_count,1)
            output=evaluation.with_name('independent_'+evaluation.name)
            output.write_text(output.read_text().replace('true','false'))
            with self.assertRaisesRegex(RuntimeError,'alterada'):
                lab.independent_monitor_batch(self.root)


if __name__=='__main__':unittest.main(verbosity=2)

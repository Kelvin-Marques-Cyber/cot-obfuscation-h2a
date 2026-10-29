"""Testes de contrato do v4 em CPU (sem pesos de modelo).

Verificam regras de correção, geração de dados, congelamento, bloqueios e
estatística. Não substituem treinos com modelos reais nem execução em CUDA.
"""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cotv4 import contrato, estatistica, runtime, tarefas  # noqa: E402

COMMIT = 'a' * 40


class ContratoTests(unittest.TestCase):
    def test_integer_contract(self):
        self.assertTrue(contrato.grade_integer('12 + 3 = 15. <answer>15</answer>', '15'))
        self.assertTrue(contrato.grade_integer('15', '15'))
        self.assertTrue(contrato.grade_integer('Conta feita.\nResposta: 15', '15'))
        self.assertFalse(contrato.grade_integer('<answer>15</answer> extra', '15'))
        self.assertFalse(contrato.grade_integer('<answer>15</answer><answer>16</answer>', '15'))
        self.assertFalse(contrato.grade_integer('A resposta é 15.', '15'))

    def test_thousands_separator_is_rejected_not_misread(self):
        for text in ('11,863', '11.863', '<answer>11,863</answer>', '<answer>11 863</answer>'):
            self.assertIsNone(contrato.parse_integer_answer(text), text)
        self.assertEqual(contrato.parse_integer_answer('<answer>11863</answer>'), 11863)
        self.assertEqual(contrato.parse_integer_answer('<answer>-42</answer>'), -42)

    def test_mcq_contract(self):
        ok = '<trace>3 * 4 = 12</trace><summary>12</summary><answer>B</answer>'
        self.assertEqual(contrato.parse_mcq(ok)['answer'], 'B')
        self.assertEqual(contrato.parse_mcq('\n' + ok + '\n')['answer'], 'B')
        self.assertIsNone(contrato.parse_mcq(ok + ' fim'))
        self.assertIsNone(contrato.parse_mcq('<trace>x</trace><answer>B</answer>'))
        self.assertIsNone(contrato.parse_mcq('<trace><trace>x</trace><summary>s</summary><answer>B</answer>'))
        self.assertIsNone(contrato.parse_mcq('<trace>x</trace><summary>s</summary><answer>E</answer>'))


class TarefasTests(unittest.TestCase):
    SPEC = {'A': {'ab': (100, 999)}, 'C': {'a': (100, 999), 'b': (12, 99), 'c': (3, 9), 'd': (100, 999)}}

    def test_sft_deterministic_disjoint_and_traces_correct(self):
        rows = tarefas.make_sft(self.SPEC, 30, 7)
        self.assertEqual(rows, tarefas.make_sft(self.SPEC, 30, 7))
        self.assertEqual(tarefas.audit(rows, 'prompt'), {'train': 60, 'dev': 60, 'test': 60})
        for row in rows:
            trace = tarefas.sft_trace(row)
            last = int(trace.rstrip('.').split('=')[-1])
            self.assertEqual(last, int(row['gold']))
            self.assertEqual(eval(row['prompt'][len('Calcule '):-1]), int(row['gold']))

    def test_calibration_items_are_excluded(self):
        calibration = tarefas.make_sft(self.SPEC, 40, 7, splits=('calibration',))
        rows = tarefas.make_sft(self.SPEC, 40, 7, exclude_prompts={r['prompt'] for r in calibration})
        self.assertFalse({r['prompt'] for r in rows} & {r['prompt'] for r in calibration})

    def test_grpo_levels_share_operand_range_and_hint_points_to_cheat(self):
        rows = tarefas.make_grpo({'operands': (11, 99)}, 6, 9)
        self.assertEqual(tarefas.audit(rows, 'question')['test'], 2 * 4 * 6)
        for row in rows:
            self.assertTrue(all(11 <= v <= 99 for v in row['parameters']))
            self.assertNotEqual(row['gold'], row['cheat'])
            self.assertEqual(row['choices']['ABCD'.index(row['gold'])], row['gold_numeric'])
            if row['level'] == 'high':
                self.assertEqual('ABCD'[eval(row['hint_expression'])], row['cheat'])
            else:
                self.assertTrue(row['hint'].endswith(row['cheat'] + '.'))
            self.assertEqual(row['structural_family'] == 'shifted', row['split'] == 'test')

    def test_hint_can_be_omitted(self):
        row = tarefas.make_grpo({'operands': (11, 99)}, 1, 9)[0]
        self.assertNotIn(row['hint'], tarefas.render_prompt(row, include_hint=False))
        self.assertIn(row['hint'], tarefas.render_prompt(row))


class EstatisticaTests(unittest.TestCase):
    def test_signflip_bootstrap_holm(self):
        self.assertAlmostEqual(estatistica.signflip([1.0] * 8), 2 / 256)
        self.assertEqual(estatistica.signflip([0.0] * 8), 1.0)
        self.assertLess(estatistica.signflip([0.1] * 20), 0.0001)
        self.assertEqual(estatistica.bootstrap_ci([0.2] * 10, iterations=500), [0.2, 0.2])
        self.assertEqual(estatistica.holm({'sft': 0.01, 'grpo': 0.04}), {'sft': True, 'grpo': True})
        self.assertEqual(estatistica.holm({'sft': 0.03, 'grpo': 0.04}), {'sft': False, 'grpo': False})
        self.assertEqual(estatistica.holm({'sft': 0.02, 'grpo': 0.06}), {'sft': True, 'grpo': False})

    def test_sft_interaction_uses_seed_pairs(self):
        group = {}
        for seed in (1, 2):
            for level in ('A', 'C'):
                for condition in ('explicit', 'direct'):
                    group[level, seed, condition] = {str(i): {'accuracy': not (level == 'C' and condition == 'direct')}
                                                     for i in range(10)}
        self.assertEqual(estatistica.paired_sft(group, [1, 2]), [-1.0, -1.0])
        del group['C', 1, 'direct']['0']
        with self.assertRaises(RuntimeError):
            estatistica.paired_sft(group, [1, 2])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name) / 'study'
        self.env = patch.dict(os.environ, {'COT_SOURCE_COMMIT': COMMIT, 'COT_TECHNICAL_TEST': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_plan_is_frozen_idempotent_and_detects_tampering(self):
        first = runtime.make_plan(self.base)
        self.assertEqual(first['datasets']['sft']['test_per_level'], {'A': 220, 'C': 220})
        self.assertEqual(first['datasets']['grpo']['test_per_level'], {'low': 240, 'high': 240})
        self.assertEqual(runtime.make_plan(self.base), first)
        self.assertEqual(len(runtime.jobs(self.base, 'sft')), 120)
        self.assertEqual(len(runtime.jobs(self.base, 'grpo')), 120)
        with (self.base / 'data' / 'sft.jsonl').open('a') as handle:
            handle.write('{}\n')
        with self.assertRaises(ValueError):
            runtime.make_plan(self.base)

    def test_changed_code_cannot_reuse_frozen_study(self):
        runtime.make_plan(self.base)
        with patch.object(runtime, 'code_hashes', return_value={'x.py': '0' * 64}):
            with self.assertRaisesRegex(ValueError, 'outro STUDY_ID'):
                runtime.make_plan(self.base)
            with self.assertRaises(RuntimeError):
                runtime.assert_frozen(self.base, 'sft')

    def write_gate(self, gate_dir, accuracies):
        gate_dir.mkdir(parents=True, exist_ok=True)
        frozen = runtime.read_json(self.base / 'protocol' / 'frozen_datasets.json')
        files = {}
        for (mode, level), accuracy in accuracies.items():
            rows = [dict(item_id=f'{mode}{level}{i}', accuracy=i < round(120 * accuracy)) for i in range(120)]
            path = gate_dir / f'dev_{mode}_{level}_base.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            files[path.name] = runtime.digest(path.read_bytes())
        manifest = dict(protocol_sha256=runtime.digest(runtime.PROTOCOL_FILE.read_bytes()),
                        code_sha256=runtime.code_hashes(), files=files,
                        datasets={m: frozen[m]['sha256'] for m in runtime.MODES})
        (gate_dir / 'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_gate_import_threshold_and_tampering(self):
        runtime.make_plan(self.base)
        gate_dir = Path(self.temp.name) / 'gate'
        passing = {('sft', 'A'): 0.9, ('sft', 'C'): 0.8, ('grpo', 'low'): 0.7, ('grpo', 'high'): 0.65}
        self.write_gate(gate_dir, passing)
        with patch.object(runtime, 'GATE_DIR', gate_dir):
            self.assertTrue(runtime.import_gate(self.base)['passed'])
            failing = dict(passing)
            failing['grpo', 'high'] = 0.64
            self.write_gate(gate_dir, failing)
            with self.assertRaisesRegex(RuntimeError, 'diverge'):
                runtime.import_gate(self.base)  # cópia local anterior não pode ser sobrescrita
        status = runtime.gate_status(self.base)
        self.assertTrue(status['passed'])
        target = self.base / 'analysis' / 'gate' / 'dev_sft_A_base.jsonl'
        target.write_text(target.read_text().replace('true', 'false', 1))
        self.assertFalse(runtime.gate_status(self.base)['passed'])

    def test_failed_gate_blocks_training_even_with_registration(self):
        runtime.make_plan(self.base)
        gate_dir = Path(self.temp.name) / 'gate'
        self.write_gate(gate_dir, {('sft', 'A'): 0.9, ('sft', 'C'): 0.2, ('grpo', 'low'): 0.9, ('grpo', 'high'): 0.9})
        with patch.object(runtime, 'GATE_DIR', gate_dir):
            self.assertFalse(runtime.import_gate(self.base)['passed'])
        runtime.declare_registration(self.base, 'https://osf.io/abcde/')
        with self.assertRaisesRegex(RuntimeError, 'triagem DEV'):
            runtime.train_batch(self.base, 'sft')

    def test_excluded_grpo_arm_is_descriptive_and_never_trains(self):
        proto = json.loads(json.dumps(runtime.protocol()))
        proto['grpo'].update(enabled=False, excluded_reason='nenhum monitor independente viável')
        proto['independent_monitor'].update(model=None, revision=None)
        with patch.object(runtime, 'protocol', return_value=proto):
            plan = runtime.make_plan(self.base)
            self.assertEqual(len(plan['model_revisions']), 3)
            gate_dir = Path(self.temp.name) / 'gate'
            self.write_gate(gate_dir, {('sft', 'A'): 0.9, ('sft', 'C'): 0.8, ('grpo', 'low'): 0.1,
                                       ('grpo', 'high'): 0.1})
            with patch.object(runtime, 'GATE_DIR', gate_dir):
                gate = runtime.import_gate(self.base)
            self.assertTrue(gate['passed'])
            self.assertFalse(gate['results']['grpo/low']['conditions_release'])
            status = runtime.study_status(self.base)
            self.assertEqual(set(status['expected']), {'sft'})
            self.assertIn('grpo', status['excluded'])
            runtime.declare_registration(self.base, 'https://osf.io/abcde/')
            with self.assertRaisesRegex(RuntimeError, 'excluído'):
                runtime.train_batch(self.base, 'grpo')
            with self.assertRaisesRegex(RuntimeError, 'excluído'):
                runtime.write_blind_sheet(self.base)
            report = runtime.final_report(self.base)
            self.assertEqual(report['study']['grpo']['status'], 'excluded_before_registration')
            self.assertNotIn('human_validation_incomplete', report['blocked_by'])
            self.assertEqual(report['verdict'], 'INSUFFICIENT_EVIDENCE')

    def test_registration_rules(self):
        runtime.make_plan(self.base)
        self.assertFalse(runtime.preregistration_gate(self.base)['passed'])
        with self.assertRaises(ValueError):
            runtime.declare_registration(self.base, 'osf.io/sem-esquema')
        run = self.base / 'runs' / 'sft' / 'x'
        run.mkdir(parents=True)
        (run / 'manifest.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'retroativo'):
            runtime.declare_registration(self.base, 'https://osf.io/abcde/')

    def test_test_split_blocked_until_all_training(self):
        runtime.make_plan(self.base)
        with self.assertRaisesRegex(RuntimeError, 'TODOS'):
            runtime.evaluate_batch(self.base, 'sft')

    def test_job_index_rejects_wrong_configuration_and_duplicates(self):
        runtime.make_plan(self.base)
        frozen = runtime.read_json(self.base / 'protocol' / 'frozen_datasets.json')['sft']
        contract = runtime.digest((self.base / 'protocol' / 'locked_protocol.json').read_bytes())
        p = runtime.protocol()

        def fake_run(name, **override):
            cfg = dict(mode='sft', model_id=p['sft']['model'], revision=p['sft']['revision'], level='A',
                       train_seed=113, condition='explicit', max_steps=120, data_sha256=frozen['sha256'],
                       experiment_contract_sha=contract)
            cfg.update(override)
            folder = self.base / 'runs' / 'sft' / name
            (folder / 'adapter').mkdir(parents=True)
            (folder / 'adapter' / 'adapter_config.json').write_text('{}')
            (folder / 'adapter' / 'adapter_model.safetensors').write_bytes(b'w')
            (folder / 'manifest.json').write_text(json.dumps(dict(run_id=name, resolved_config=cfg)))
            (folder / 'status.json').write_text(json.dumps(dict(
                status='trained', global_step=cfg['max_steps'],
                adapter_sha256={'adapter_model.safetensors': runtime.digest(b'w')})))
            return folder
        fake_run('ok')
        self.assertEqual(len(runtime.job_index(self.base, 'sft')), 1)
        fake_run('dup')
        with self.assertRaisesRegex(RuntimeError, 'duplicado'):
            runtime.job_index(self.base, 'sft')
        import shutil
        shutil.rmtree(self.base / 'runs' / 'sft' / 'dup')
        bad = fake_run('bad', max_steps=60, train_seed=114)
        with self.assertRaisesRegex(RuntimeError, 'incompatível'):
            runtime.job_index(self.base, 'sft')
        shutil.rmtree(bad)
        (self.base / 'runs' / 'sft' / 'ok' / 'adapter' / 'adapter_model.safetensors').write_bytes(b'x')
        with self.assertRaisesRegex(RuntimeError, 'alterado'):
            runtime.job_index(self.base, 'sft')

    def test_report_is_insufficient_without_data_and_humans(self):
        runtime.make_plan(self.base)
        report = runtime.final_report(self.base)
        self.assertEqual(report['verdict'], 'INSUFFICIENT_EVIDENCE')
        self.assertIn('incomplete_sft', report['blocked_by'])
        self.assertIn('human_validation_incomplete', report['blocked_by'])
        self.assertFalse(report['strategic_intent_proven'])

    def test_human_and_monitor_gates_require_real_complete_inputs(self):
        human = self.base / 'human'
        human.mkdir(parents=True)
        self.assertFalse(runtime.human_quality_gate(self.base)['passed'])
        with (human / 'blinded_items.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['annotation_id'])
            writer.writeheader()
            writer.writerows({'annotation_id': str(i)} for i in range(360))
        with (human / 'completed_annotations.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['annotation_id', 'rater', 'label'])
            writer.writeheader()
            writer.writerows({'annotation_id': str(i), 'rater': 'r1', 'label': '1'} for i in range(360))
        self.assertEqual(runtime.human_quality_gate(self.base)['reason'], 'all_items_require_two_humans')
        p = runtime.protocol()['independent_monitor']
        rows = [dict(label=str(i % 2), flag='1', split='heldout', source='human', monitor_model=p['model'],
                     monitor_revision=p['revision']) for i in range(200)]
        with (human / 'independent_monitor_scored.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = runtime.monitor_gate(self.base)
        self.assertFalse(result['passed'])
        self.assertEqual((result['tpr'], result['fpr']), (1, 1))

    def test_technical_test_is_separate_and_never_scientific(self):
        with patch.dict(os.environ, {'COT_TECHNICAL_TEST': '1'}):
            plan = runtime.make_plan(self.base)
            self.assertTrue(plan['protocol']['technical_test'])
            self.assertEqual(len(runtime.jobs(self.base, 'sft')), 6)
            report = runtime.final_report(self.base)
            self.assertIn('technical_test_not_scientific_evidence', report['blocked_by'])
            self.assertEqual(report['verdict'], 'INSUFFICIENT_EVIDENCE')
        with self.assertRaises(ValueError):
            runtime.make_plan(self.base)  # o mesmo diretório não vira estudo científico


if __name__ == '__main__':
    unittest.main(verbosity=2)


class SelecaoTriagemTests(unittest.TestCase):
    def test_grpo_gate_covers_all_four_domains(self):
        from cotv4.avaliacao import select_dev
        rows = tarefas.make_grpo({'operands': (11, 99)}, 60, 5)
        for level in ('low', 'high'):
            selected = select_dev('grpo', rows, level)
            self.assertEqual(len(selected), 120)
            self.assertEqual({d: sum(r['domain'] == d for r in selected) for d in tarefas.DOMAINS},
                             {d: 30 for d in tarefas.DOMAINS})
            self.assertTrue(all(r['split'] == 'dev' and r['level'] == level for r in selected))

    def test_sft_gate_uses_first_dev_items(self):
        from cotv4.avaliacao import select_dev
        rows = tarefas.make_sft(TarefasTests.SPEC, 130, 5)
        selected = select_dev('sft', rows, 'C')
        self.assertEqual(selected, [r for r in rows if r['split'] == 'dev' and r['level'] == 'C'][:120])

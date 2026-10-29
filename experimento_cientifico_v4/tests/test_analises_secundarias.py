"""Análises secundárias declaradas no registro: contas conferidas em dados sintéticos."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analises_secundarias as sec  # noqa: E402
from cotv4 import runtime  # noqa: E402

SEEDS = list(range(113, 133))


def fake_rows(per_level=10):
    rows = []
    for level in ('A', 'C'):
        for split in ('train', 'dev', 'test'):
            rows += [dict(item_id=f'{level}-{split}-{i}', level=level, split=split) for i in range(per_level)]
    return rows


def fake_group(rows, correct):
    """correct(level, condition, index) -> bool; o mesmo padrão em todas as sementes."""
    group = {}
    for level in ('A', 'C'):
        test = [r for r in rows if r['split'] == 'test' and r['level'] == level]
        for seed in SEEDS:
            for condition in ('explicit', 'direct', 'generic'):
                group[level, seed, condition] = {
                    r['item_id']: dict(accuracy=correct(level, condition, i), format_valid=True,
                                       finish_reason='length' if i == 0 else 'eos')
                    for i, r in enumerate(test)}
    return group


class SecondaryAnalyses(unittest.TestCase):
    def setUp(self):
        self.rows = fake_rows()

    def test_exposed_items_are_first_test_items_of_each_level(self):
        exposed = sec.exposed_items(self.rows, 4)
        self.assertEqual(exposed['A'], [f'A-test-{i}' for i in range(4)])
        self.assertEqual(exposed['C'], [f'C-test-{i}' for i in range(4)])

    def test_interaction_matches_primary_and_excludes_items(self):
        # direct erra os 4 primeiros itens de C; explicit acerta tudo.
        group = fake_group(self.rows, lambda lv, c, i: not (c == 'direct' and lv == 'C' and i < 4))
        from cotv4 import estatistica
        full = sec.interaction(group, SEEDS, 'direct', 'explicit')
        self.assertEqual(full, estatistica.paired_sft(group, SEEDS))
        self.assertAlmostEqual(full[0], -0.4)
        keep = {lv: {f'{lv}-test-{i}' for i in range(4, 10)} for lv in ('A', 'C')}
        self.assertEqual(sec.interaction(group, SEEDS, 'direct', 'explicit', keep), [0.0] * 20)

    def test_analyse_reports_all_blocks_without_touching_verdict(self):
        group = fake_group(self.rows, lambda lv, c, i: (c != 'direct' or lv == 'A' or i % 2 == 0))
        protocol = dict(runtime.protocol())
        with patch.object(runtime, 'load_primary', return_value=group), \
                patch.object(runtime, 'seeds', return_value=SEEDS), \
                patch.object(runtime, 'read_jsonl', return_value=self.rows), \
                patch.object(runtime, 'protocol', return_value=protocol):
            result = sec.analyse(Path('.'))
        self.assertFalse(result['confirmatory'])
        s1 = result['S1_sem_itens_do_teste_tecnico']
        self.assertEqual(s1['items_per_level'], {'A': 6, 'C': 6})
        self.assertAlmostEqual(s1['mean'], -0.5)
        self.assertAlmostEqual(result['S2_acuracia']['C/direct']['mean'], 0.5)
        self.assertAlmostEqual(result['S3_generic']['generic_menos_explicit']['mean'], 0.0)
        self.assertAlmostEqual(result['S3_generic']['direct_menos_generic']['mean'], -0.5)
        self.assertAlmostEqual(result['S4_truncamento']['A/explicit']['mean'], 0.1)
        self.assertLess(s1['signflip_p'], 0.001)


if __name__ == '__main__':
    unittest.main()

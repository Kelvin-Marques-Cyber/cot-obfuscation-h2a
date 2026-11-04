"""Seleção de sementes e descritivos da análise S5 (DESVIO_01.md)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analise_desvio_01 as s5  # noqa: E402


def full(seeds, affected=None):
    affected = affected or {}
    return {(lv, seed, cond): affected.get((lv, seed, cond), [])
            for lv in s5.LEVELS for seed in seeds for cond in s5.CONDITIONS}


class CleanSeeds(unittest.TestCase):
    def test_seed_with_any_skipped_adapter_is_excluded(self):
        skips = full([1, 2, 3], {('A', 2, 'generic'): [15], ('C', 3, 'direct'): [41, 60]})
        self.assertEqual(s5.clean_seeds(skips, [1, 2, 3]), [1])

    def test_no_skips_keeps_all_seeds_in_order(self):
        self.assertEqual(s5.clean_seeds(full([5, 4, 6]), [5, 4, 6]), [5, 4, 6])

    def test_descriptives_count_runs_and_steps(self):
        d = s5.descriptives(full([1, 2], {('A', 2, 'generic'): [15], ('A', 1, 'generic'): [3, 9]}))
        self.assertEqual(d['runs_with_skips'], 2)
        self.assertEqual(d['skipped_steps_total'], 3)
        self.assertEqual(d['per_cell']['A/generic'],
                         dict(runs=2, runs_with_skips=2, skipped_steps_total=3, max_skipped_in_one_run=2))
        self.assertEqual(d['per_cell']['C/direct']['runs_with_skips'], 0)

    def test_skipped_by_job_reads_status(self):
        root = Path(tempfile.mkdtemp())
        (root / 'r1').mkdir()
        (root / 'r2').mkdir()
        (root / 'r1' / 'status.json').write_text(json.dumps(dict(status='trained', skipped_fp16_steps=[41])))
        (root / 'r2' / 'status.json').write_text(json.dumps(dict(status='trained')))
        out = s5.skipped_by_job({('A', 1, 'direct'): root / 'r1', ('C', 1, 'direct'): root / 'r2'})
        self.assertEqual(out, {('A', 1, 'direct'): [41], ('C', 1, 'direct'): []})


if __name__ == '__main__':
    unittest.main()

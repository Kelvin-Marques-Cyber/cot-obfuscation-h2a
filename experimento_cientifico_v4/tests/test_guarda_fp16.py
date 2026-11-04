"""Guarda de valores não finitos do treino (desvio 1, DESVIO_01.md).

Em fp16, um passo descartado pelo escalonador de gradientes registra norma nan e
não interrompe o treino; os demais casos não finitos continuam interrompendo.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import transformers  # noqa: F401
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


@unittest.skipUnless(HAVE_STACK, 'requer transformers')
class FiniteGuard(unittest.TestCase):
    def setUp(self):
        from cotv4 import treino
        self.treino = treino
        self.dir = Path(tempfile.mkdtemp())
        self.guard = treino._finite_callback(self.dir, 120)

    def step(self, n, skipped, fp16=True):
        args, state = SimpleNamespace(fp16=fp16), SimpleNamespace(global_step=n)
        self.guard.on_step_end(args, state, None, optimizer=SimpleNamespace(step_was_skipped=skipped))
        return args, state

    def test_skipped_step_with_nan_grad_norm_continues_and_is_recorded(self):
        args, state = self.step(15, True)
        self.guard.on_log(args, state, None, logs={'loss': 2.1, 'grad_norm': float('nan')})
        self.assertEqual(json.loads((self.dir / self.treino.SKIPPED_FILE).read_text()), [15])
        self.assertEqual(self.treino.skipped_steps(self.dir), [15])

    def test_nan_grad_norm_without_skip_stops(self):
        args, state = self.step(16, False)
        with self.assertRaises(FloatingPointError):
            self.guard.on_log(args, state, None, logs={'loss': 2.1, 'grad_norm': float('nan')})

    def test_non_finite_loss_stops_even_when_step_was_skipped(self):
        args, state = self.step(20, True)
        with self.assertRaises(FloatingPointError):
            self.guard.on_log(args, state, None, logs={'loss': float('inf'), 'grad_norm': float('nan')})

    def test_more_than_ten_percent_skipped_stops(self):
        for n in range(1, 13):
            self.step(n, True)
        with self.assertRaises(FloatingPointError):
            self.step(13, True)

    def test_fp32_keeps_original_behaviour(self):
        args, state = self.step(5, False, fp16=False)
        with self.assertRaises(FloatingPointError):
            self.guard.on_log(args, state, None, logs={'loss': 1.0, 'grad_norm': float('inf')})

    def test_finite_logs_pass_untouched(self):
        args, state = self.step(5, False)
        self.guard.on_log(args, state, None, logs={'loss': 1.0, 'grad_norm': 0.5})
        self.assertFalse((self.dir / self.treino.SKIPPED_FILE).exists())

    def test_non_finite_adapter_is_refused(self):
        import torch
        from safetensors.torch import save_file
        save_file({'w': torch.tensor([1.0, float('nan')])}, str(self.dir / 'adapter_model.safetensors'))
        with self.assertRaises(FloatingPointError):
            self.treino.check_finite_adapter(self.dir)
        save_file({'w': torch.tensor([1.0, 2.0])}, str(self.dir / 'adapter_model.safetensors'))
        self.treino.check_finite_adapter(self.dir)


if __name__ == '__main__':
    unittest.main()

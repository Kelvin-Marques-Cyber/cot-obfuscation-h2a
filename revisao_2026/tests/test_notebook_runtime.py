import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from notebook_runtime import run_logged


class NotebookRuntimeTests(unittest.TestCase):
    def test_stdout_and_stderr_are_visible_and_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "logs" / "step.log"
            visible = io.StringIO()
            with contextlib.redirect_stdout(visible):
                result = run_logged(
                    [sys.executable, "-u", "-c",
                     "import sys; print(sys.argv[1]); print('diagnóstico', file=sys.stderr)",
                     "argumento com espaços"], log)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(log.read_text(encoding="utf-8"), visible.getvalue())
            self.assertIn("argumento com espaços", visible.getvalue())
            self.assertIn("diagnóstico", visible.getvalue())

    def test_exception_includes_real_error_and_log_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "failure.log"
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError) as caught:
                    run_logged([sys.executable, "-c",
                                "raise ModuleNotFoundError('pacote_ausente')"], log)
            error = str(caught.exception)
            self.assertIn("código 1", error)
            self.assertIn("ModuleNotFoundError: pacote_ausente", error)
            self.assertIn(str(log), error)
            self.assertIn("ModuleNotFoundError: pacote_ausente", log.read_text())

    def test_runs_from_requested_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                run_logged([sys.executable, "-c", "import os; print(os.getcwd())"],
                           path / "cwd.log", cwd=path)
            self.assertEqual((path / "cwd.log").read_text().strip(), str(path.resolve()))


if __name__ == "__main__":
    unittest.main()

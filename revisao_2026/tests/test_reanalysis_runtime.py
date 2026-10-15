import contextlib
import importlib.metadata as metadata
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import reanalyse


class ReanalysisRuntimeTests(unittest.TestCase):
    def test_complete_analysis_without_pdf_packages(self):
        """A ausência de pacotes de PDF não interrompe cálculos nem figuras."""
        installed_version = metadata.version

        def without_pdf(name):
            if name in ("reportlab", "pypdf"):
                raise metadata.PackageNotFoundError(name)
            return installed_version(name)

        with tempfile.TemporaryDirectory() as tmp:
            out, figures = Path(tmp) / "results", Path(tmp) / "figures"
            with patch.object(reanalyse, "OUT", out), \
                 patch.object(reanalyse, "FIG", figures), \
                 patch.object(metadata, "version", side_effect=without_pdf), \
                 contextlib.redirect_stdout(io.StringIO()):
                reanalyse.main()
            report = json.loads((out / "report.json").read_text())
            self.assertEqual(report["n_rows"], 540)
            self.assertTrue(report["all_aggregates_match"])
            self.assertEqual(report["original_false_accepts"], 1250)
            self.assertEqual(report["strict_false_accepts"], 0)
            self.assertEqual(report["disjoint_generated_items"], 4500)
            self.assertIsNone(report["environment"]["packages"]["reportlab"])
            self.assertIsNone(report["environment"]["packages"]["pypdf"])
            self.assertEqual(len(list(figures.glob("*.png"))), 3)


if __name__ == "__main__":
    unittest.main()

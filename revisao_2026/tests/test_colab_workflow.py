import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from colab_runtime import ColabSession
from gpu_lab.notebook_pipeline import prepare,preserve_dataset,paths
from gpu_lab.local_monitor import choose_flag
from gpu_lab.analysis import export_annotations
from instruments import make_disjoint_data


def fake_revision(model,revision="main"):
    return hashlib.sha1(model.encode()).hexdigest()


class ColabWorkflowTests(unittest.TestCase):
    def test_invalid_colab_backend_is_replaced_only_for_child_processes(self):
        """O kernel mantém o backend inline, mas subprocessos usam Agg."""
        inherited = "module://matplotlib_inline.backend_inline"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"MPLBACKEND": inherited}):
            session = ColabSession(Path(__file__).resolve().parents[1], Path(tmp))
            child_env = session.environment()
            self.assertEqual(child_env["MPLBACKEND"], "Agg")
            self.assertEqual(os.environ["MPLBACKEND"], inherited)
            probe = subprocess.run(
                [sys.executable, "-c", "import matplotlib; print(matplotlib.get_backend())"],
                env=child_env, capture_output=True, text=True, check=True,
            )
            self.assertEqual(probe.stdout.strip().lower(), "agg")


    def test_quick_plan_does_not_require_4bit_quantization(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for mode in ("sft","grpo"):
                plan,_=prepare(tmp,"quick",mode,resolver=fake_revision)
                self.assertTrue(plan)
                self.assertTrue(all(cfg["load_in_4bit"] is False for cfg in plan))

    def test_legacy_quick_migration_preserves_model_revisions(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            plan,p=prepare(tmp,"quick","sft",resolver=fake_revision)
            frozen=p["runs"]/"frozen_plan.json"
            old=[dict(cfg,load_in_4bit=True) for cfg in plan]
            frozen.write_text(json.dumps(old))
            with patch("gpu_lab.notebook_pipeline.resolve_revision",
                       side_effect=AssertionError("Nao deve consultar rede")):
                migrated,_=prepare(tmp,"quick","sft",resolver=lambda *_:self.fail("sem rede"))
            self.assertEqual([c["revision"] for c in migrated],[c["revision"] for c in old])
            self.assertTrue(all(c["load_in_4bit"] is False for c in migrated))
            self.assertEqual(json.loads((p["runs"]/"frozen_plan_4bit_backup.json").read_text()),old)

    def test_completed_legacy_plan_is_preserved_and_fp16_runs_separately(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            plan,p=prepare(tmp,"quick","sft",resolver=fake_revision)
            frozen=p["runs"]/"frozen_plan.json"
            old=[dict(cfg,load_in_4bit=True) for cfg in plan]
            frozen.write_text(json.dumps(old))
            trained=p["runs"]/"previous-run"
            trained.mkdir()
            (trained/"status.json").write_text(json.dumps({"status":"trained"}))
            migrated,new_paths=prepare(tmp,"quick","sft",resolver=lambda *_:self.fail("sem rede"))
            self.assertEqual(json.loads(frozen.read_text()),old)
            self.assertTrue((trained/"status.json").is_file())
            self.assertNotEqual(p["runs"],new_paths["runs"])
            self.assertTrue(all(cfg["load_in_4bit"] is False for cfg in migrated))
            self.assertEqual([cfg["revision"] for cfg in old],[cfg["revision"] for cfg in migrated])
            self.assertEqual(paths(tmp,"quick","sft")["runs"],new_paths["runs"])
            again,again_paths=prepare(tmp,"quick","sft",resolver=lambda *_:self.fail("sem rede"))
            self.assertEqual(again,migrated)
            self.assertEqual(again_paths,new_paths)

    def test_profile_is_forwarded_to_gpu_child_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            session=ColabSession(Path(__file__).resolve().parents[1],Path(tmp),
                                 profile="quick",gpu=True)
            self.assertEqual(session.environment()["COT_PROFILE"],"quick")

    def test_quick_runs_all_three_conditions_without_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            plan,p=prepare(tmp,"quick","grpo",resolver=fake_revision)
            self.assertEqual({c["condition"] for c in plan},{"none","cot_monitor","summary_monitor"})
            self.assertTrue(all(c["monitor_backend"]=="local" for c in plan))
            self.assertTrue(p["data"].exists())
            self.assertTrue(all(len(c["revision"])==40 and len(c["monitor_revision"])==40 for c in plan))

    def test_rerun_reuses_frozen_model_and_monitor_revisions(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            first,_=prepare(tmp,"quick","grpo",resolver=fake_revision)
            with patch("gpu_lab.notebook_pipeline.resolve_revision",side_effect=AssertionError("network")):
                second,_=prepare(tmp,"quick","grpo",resolver=lambda *_:self.fail("Must reuse revisions"))
            self.assertEqual(first,second)

    def test_sft_conditions_and_profiles_have_separate_storage(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            quick,a=prepare(tmp,"quick","sft",resolver=fake_revision)
            pilot,b=prepare(tmp,"pilot","sft",resolver=fake_revision)
            self.assertEqual(len(quick),3)
            self.assertEqual(len(pilot),27)
            self.assertNotEqual(a["data"],b["data"])
            self.assertEqual({c["condition"] for c in quick},{"explicit","direct","generic"})

    def test_existing_dataset_never_overwritten_by_different_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.jsonl"
            preserve_dataset(path,make_disjoint_data(1))
            original=path.read_bytes()
            with self.assertRaisesRegex(ValueError,"Dados existentes"):
                preserve_dataset(path,make_disjoint_data(2))
            self.assertEqual(path.read_bytes(),original)

    def test_invalid_monitor_scores_do_not_become_negative_labels(self):
        for scores in ([float("nan"),0],[1],[-1,float("inf")]):
            with self.subTest(scores=scores),self.assertRaises(ValueError):
                choose_flag(scores)
        self.assertTrue(choose_flag([-4,-1]))
        self.assertFalse(choose_flag([-1,-4]))

    def test_annotations_distinguish_base_and_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"runs"
            root.mkdir()
            row=dict(run_id="run",model_id="model",condition="none",level="low",item_id="item",
                evaluation_mode="full",total_response_budget=128,prompt="question",accuracy=True,
                exploit=False,generation="<trace>x</trace><summary>y</summary><answer>A</answer>")
            (root/"evaluation_test.jsonl").write_text("".join(json.dumps(dict(row,base_only=b))+"\n"
                                                             for b in (True,False)))
            destination=Path(tmp)/"annotation"
            self.assertEqual(export_annotations(root,destination),2)
            import csv
            with (destination/"blinded_annotations.csv").open(newline="") as f:
                records=list(csv.DictReader(f))
            self.assertEqual(len({r["annotation_id"] for r in records}),2)

    def test_bundle_excludes_checkpoints_weights_and_cache(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            storage=Path(tmp)/"storage"
            run=storage/"quick/runs/grpo/a"
            (run/"checkpoints").mkdir(parents=True)
            (run/"adapter").mkdir()
            (run/"checkpoints/state.json").write_text("{}")
            (run/"adapter/config.json").write_text("{}")
            (run/"manifest.json").write_text("{}")
            (run/"weights.safetensors").write_bytes(b"weights")
            (run/"cache.sqlite").write_bytes(b"cache")
            session=ColabSession(Path(__file__).resolve().parents[1],storage)
            with contextlib.redirect_stdout(io.StringIO()):
                output=session.bundle("grpo")
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(archive.namelist(),["quick/runs/grpo/a/manifest.json"])


if __name__=="__main__":
    unittest.main()

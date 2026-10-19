"""Regressões de sessões reaproveitadas e ambientes parcialmente danificados."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from colab_runtime import ColabSession


class ColabSessionRecoveryTests(unittest.TestCase):
    def test_bootstrap_loads_both_modules_from_new_source_in_same_kernel(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for name in ("old","new"):
                source=base/name/"src"
                source.mkdir(parents=True)
                (source/"notebook_runtime.py").write_text(f"MARKER={name!r}\n")
                (source/"colab_runtime.py").write_text(
                    "from notebook_runtime import MARKER\n"
                    "class ColabSession:\n"
                    "    def __init__(self,*args,**kwargs): self.marker=MARKER\n")
            script="""
from pathlib import Path
import os
from create_notebooks import BOOTSTRAP
base=Path(os.environ['COT_SESSION_TEST_ROOT'])
namespace={'__name__':'__main__'}
os.environ['COT_STORAGE_ROOT']=str(base/'results')
for version in ['old','new']:
    os.environ['COT_PROJECT_ROOT']=str(base/version)
    source=BOOTSTRAP.replace('__SOURCE_REF__','a'*40).replace('__NEEDS_GPU__','False')
    exec(compile(source,'bootstrap','exec'),namespace)
    assert namespace['SESSION'].marker==version
print('both sources verified')
"""
            env=dict(os.environ,PYTHONPATH=str(ROOT/"src"),COT_SESSION_TEST_ROOT=tmp)
            result=subprocess.run([sys.executable,"-c",script],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("both sources verified",result.stdout)

    @unittest.skipUnless(importlib.util.find_spec("ensurepip"),"Python sem ensurepip para fixture offline")
    def test_cached_environment_is_checked_and_repaired_without_touching_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            project=base/"project"
            (project/"src").mkdir(parents=True)
            (project/"src/cpu_preflight.py").write_text("import cot_env_probe; assert cot_env_probe.VALUE==7\n")
            wheel=base/"cot_env_probe-1.0-py3-none-any.whl"
            files={"cot_env_probe.py":"VALUE=7\n",
                "cot_env_probe-1.0.dist-info/METADATA":"Metadata-Version: 2.1\nName: cot-env-probe\nVersion: 1.0\n",
                "cot_env_probe-1.0.dist-info/WHEEL":"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
            files["cot_env_probe-1.0.dist-info/RECORD"]="".join(name+",,\n" for name in [*files,"cot_env_probe-1.0.dist-info/RECORD"])
            with zipfile.ZipFile(wheel,"w") as archive:
                for name,content in files.items():
                    archive.writestr(name,content)
            (project/"requirements.lock.txt").write_text("--no-index\n"+str(wheel)+"\n")
            storage=base/"results"
            storage.mkdir()
            checkpoint=storage/"checkpoint.json"
            checkpoint.write_text('{"keep":true}')
            with contextlib.redirect_stdout(io.StringIO()):
                first=ColabSession(project,storage,cache=base/"environments").ensure_environment()
            module=Path(subprocess.check_output([str(first.python),"-c",
                "import cot_env_probe;print(cot_env_probe.__file__)"],text=True).strip())
            module.unlink()
            self.assertTrue((first.python.parent.parent/"cot-ready.json").is_file())
            # Mesmo com o marcador antigo, a importação quebrada é detectada.
            with contextlib.redirect_stdout(io.StringIO()):
                second=ColabSession(project,storage,cache=base/"environments").ensure_environment()
            self.assertEqual(first.python,second.python)
            probe=subprocess.check_output([str(second.python),"-c","import cot_env_probe;print(cot_env_probe.VALUE)"],text=True)
            self.assertEqual(probe.strip(),"7")
            self.assertEqual(checkpoint.read_text(),'{"keep":true}')


if __name__=="__main__":
    unittest.main()

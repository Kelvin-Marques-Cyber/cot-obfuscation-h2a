"""Preparação automática do laboratório, sem importar bibliotecas no kernel."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from notebook_runtime import run_logged


class ColabSession:
    def __init__(self, root, storage, profile="quick", gpu=False, cache=None):
        if profile not in ("quick", "pilot", "full"):
            raise ValueError("Perfil inválido")
        self.root = Path(root).resolve()
        self.storage = Path(storage).resolve()
        self.profile = profile
        self.gpu = gpu
        self.cache = Path(cache or os.environ.get("COT_ENV_CACHE", "/content/cot-lab/envs"
            if Path("/content").is_dir() else Path.home()/".cache/cot-lab/envs")).resolve()
        self.logs = self.storage / "logs" / profile
        self.logs.mkdir(parents=True, exist_ok=True)
        self.python = None

    def environment(self):
        env = dict(os.environ)
        env.update(PYTHONPATH=str(self.root/"src")+os.pathsep+env.get("PYTHONPATH", ""),
                   PYTHONUNBUFFERED="1", PIP_DISABLE_PIP_VERSION_CHECK="1",
                   TOKENIZERS_PARALLELISM="false", USE_TF="0", USE_FLAX="0",
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   COT_TEXT_ONLY="1", HF_HUB_DISABLE_TELEMETRY="1")
        return env

    def ensure_environment(self):
        torch_version = "none"
        torch_site = None
        if self.gpu:
            result = subprocess.run([sys.executable, "-c", "import json,torch; from pathlib import Path; "
                "print(json.dumps({'available':torch.cuda.is_available(),'version':torch.__version__,"
                "'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"
                "'site':str(Path(torch.__file__).resolve().parent.parent)}))"],
                capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("Este notebook precisa de PyTorch e GPU. No Colab, selecione "
                                   "Ambiente de execução > Alterar tipo de ambiente > GPU T4.\n"+result.stderr)
            info = json.loads(result.stdout.strip().splitlines()[-1])
            if not info["available"]:
                raise RuntimeError("Selecione GPU T4 em Ambiente de execução > Alterar tipo de ambiente. "
                                   "O notebook 01 funciona em CPU; os demais executam modelos.")
            torch_version = info["version"]
            torch_site = info["site"]
            print("GPU:", info["name"], "| PyTorch:", torch_version, flush=True)
        requirements = self.root / ("requirements-gpu.txt" if self.gpu else "requirements.lock.txt")
        signature = hashlib.sha256((requirements.read_text()+sys.version+torch_version).encode()).hexdigest()[:16]
        target = self.cache / ("gpu-" if self.gpu else "cpu-")
        target = target.with_name(target.name+signature)
        python = target / ("Scripts/python.exe" if os.name=="nt" else "bin/python")
        if not python.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run([sys.executable, "-m", "venv", *(["--system-site-packages"]
                if self.gpu else []), str(target)], capture_output=True, text=True)
            if result.returncode:
                # Colab pode não distribuir ensurepip. Bootstrap isolado em --target.
                if target.exists():
                    shutil.rmtree(target)
                bootstrap = self.cache / "virtualenv-bootstrap"
                run_logged([sys.executable,"-m","pip","install","--quiet","--target",str(bootstrap),
                            "virtualenv==20.35.3"], self.logs/"virtualenv_install.log")
                env = self.environment()
                env["PYTHONPATH"] = str(bootstrap)+os.pathsep+env["PYTHONPATH"]
                run_logged([sys.executable,"-m","virtualenv", *(["--system-site-packages"]
                    if self.gpu else []),str(target)], self.logs/"virtualenv_create.log",env=env)
        self.python = python
        if torch_site:
            site_paths=json.loads(subprocess.check_output([str(python),"-c",
                "import json,site; print(json.dumps(site.getsitepackages()))"],text=True))
            own=next(Path(p) for p in site_paths if Path(p).is_relative_to(target))
            (own/"cot_host_torch.pth").write_text(torch_site+"\n")
        ready = target / "cot-ready.json"
        if not ready.is_file():
            self.command(["-m","pip","install","-r",str(requirements)], "dependencies.log")
            self.command(["-m","gpu_lab.preflight" if self.gpu else "cpu_preflight"], "environment_check.log")
            ready.write_text(json.dumps({"requirements_sha256":signature,"python":sys.version,
                                         "torch":torch_version}, indent=2))
        print("Ambiente pronto. Resultados:", self.storage, flush=True)
        return self

    def command(self, args, log_name):
        if self.python is None:
            raise RuntimeError("Prepare o ambiente primeiro")
        return run_logged([str(self.python),"-u",*map(str,args)], self.logs/log_name,
                          cwd=self.root, env=self.environment())

    def pipeline(self, stage, mode="grpo"):
        return self.command(["-m","gpu_lab.notebook_pipeline",stage,"--mode",mode,
            "--storage",str(self.storage),"--profile",self.profile],f"{mode}_{stage}.log")

    def show_csv(self, path):
        from IPython.display import HTML, display
        path = Path(path)
        if not path.is_file():
            print("Ainda não disponível:",path.name)
            return
        with path.open(encoding="utf-8",newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return
        header = "".join("<th>"+html.escape(v)+"</th>" for v in rows[0])
        body = "".join("<tr>"+"".join("<td>"+html.escape(v)+"</td>" for v in row)+"</tr>"
                       for row in rows[1:101])
        display(HTML("<h3>"+html.escape(path.name)+"</h3><table><thead><tr>"+header+
            "</tr></thead><tbody>"+body+"</tbody></table>"))

    def bundle(self, mode):
        import zipfile
        destination = self.storage/"downloads"/f"resultados_{mode}_{self.profile}.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        folders = ([self.storage/"reanalysis"] if mode=="reanalysis" else
                   [self.storage/self.profile/"runs"/mode,
                    self.storage/self.profile/"analysis"/mode,
                    self.storage/self.profile/"annotation"/mode,
                    self.storage/self.profile/"data"])
        folders.append(self.logs)
        with zipfile.ZipFile(destination,"w",zipfile.ZIP_DEFLATED) as archive:
            for folder in folders:
                if folder.exists():
                    for path in sorted(folder.rglob("*")):
                        if (path.is_file() and "checkpoints" not in path.parts and
                            "adapter" not in path.parts and path.suffix in
                            (".json",".jsonl",".csv",".txt",".log",".png",".html")):
                            archive.write(path,path.relative_to(self.storage))
        print("ZIP salvo:",destination,flush=True)
        try:
            from google.colab import files
        except ImportError:
            return destination
        files.download(str(destination))
        return destination

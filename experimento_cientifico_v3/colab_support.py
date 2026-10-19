"""Colab entry points for the prospective study; scientific choices stay locked."""
from pathlib import Path
import hashlib
import importlib
import json
import os
import subprocess
import sys
import zipfile


class ScientificSession:
    def __init__(self, repo, storage, source_commit, script_sha256, on_colab=False,
                 storage_persistent=True, storage_backend='local'):
        self.repo=Path(repo).resolve()
        self.source=self.repo/'revisao_2026'
        self.package=self.repo/'experimento_cientifico_v3'
        self.script=self.package/'scientific_runtime.py'
        if hashlib.sha256(self.script.read_bytes()).hexdigest()!=script_sha256:
            raise RuntimeError('O código científico diverge do hash do notebook. Reabra o link atualizado.')
        self.storage=Path(storage).resolve()
        self.storage.mkdir(parents=True,exist_ok=True)
        self.on_colab=on_colab
        self.storage_persistent=storage_persistent
        self.storage_backend=storage_backend
        os.environ.update(COT_SOURCE_COMMIT=source_commit,COT_SOURCE_DIR=str(self.source),
                          COT_TEXT_ONLY='1',MPLBACKEND='Agg',USE_TF='0',USE_FLAX='0')
        sys.path.insert(0,str(self.source/'src'))
        for name in ('colab_runtime','notebook_runtime'):
            sys.modules.pop(name,None)
        importlib.invalidate_caches()
        module=importlib.import_module('colab_runtime')
        if Path(module.__file__).resolve()!=self.source/'src/colab_runtime.py':
            raise RuntimeError('Outra versão do ambiente está carregada na sessão.')
        self.session_type=module.ColabSession
        self.gpu_session=None

    def gpu(self):
        if self.on_colab and not self.storage_persistent:
            print('Treinos pausados: conecte o Drive e reexecute desde a primeira célula.',flush=True)
            return None
        if self.gpu_session is None:
            # "quick" selects the tested fp16 dependency stack, not the study size.
            self.gpu_session=self.session_type(self.source,self.storage,profile='quick',gpu=True)
            try:
                self.gpu_session.ensure_environment()
            except Exception:
                self.gpu_session=None
                self.failure_export()
                raise
        return self.gpu_session

    def execute(self, stage, mode='sft', max_jobs=2, gpu=False):
        if self.on_colab and not self.storage_persistent:
            if stage in ('dev','train','evaluate','monitor','controls'):
                print('Etapa aguardando armazenamento persistente:',stage,
                      '| conecte o Drive e execute novamente.',flush=True)
                return dict(status='waiting_for_persistent_storage',stage=stage,
                            completed=0,total=120,executed=[])
            gpu=False
        operation=self.storage/'operations'/f'{stage}_{mode}.json'
        operation.parent.mkdir(parents=True,exist_ok=True)
        operation.unlink(missing_ok=True)
        args=[str(self.script),stage,'--storage',str(self.storage),'--mode',mode,
              '--max-jobs',str(max_jobs),'--result-json',str(operation)]
        try:
            if gpu:
                self.gpu().command(args,f'scientific_{stage}_{mode}.log')
            else:
                if stage=='models':
                    check=subprocess.run([sys.executable,'-c','import huggingface_hub'],
                                         capture_output=True)
                    if check.returncode:
                        subprocess.run([sys.executable,'-m','pip','install','-q',
                                        'huggingface_hub==0.36.2'],check=True)
                from notebook_runtime import run_logged
                env=dict(os.environ,PYTHONPATH=str(self.source/'src')+os.pathsep+
                         os.environ.get('PYTHONPATH',''),PYTHONUNBUFFERED='1')
                run_logged([sys.executable,'-u',*args],
                           self.storage/'logs'/f'scientific_{stage}_{mode}.log',env=env)
        except Exception:
            self.failure_export()
            raise
        return json.loads(operation.read_text(encoding='utf-8'))

    def export(self, registration=False, download=True):
        name='pacote_para_pre_registro.zip' if registration else 'resultados_cientificos_v3.zip'
        target=self.storage/'downloads'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        folders=('protocol',) if registration else ('protocol','operations','logs','analysis','human','runs')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
            for folder in folders:
                for path in sorted((self.storage/folder).rglob('*')):
                    if path.is_file() and not {'adapter','checkpoints'}&set(path.relative_to(self.storage).parts):
                        if path.suffix in ('.json','.jsonl','.csv','.md','.log','.txt'):
                            archive.write(path,path.relative_to(self.storage))
            archive.write(self.package/'PROTOCOLO_PRE_REGISTRO_v3.md','PROTOCOLO_PRE_REGISTRO_v3.md')
        print('ZIP salvo:',target,flush=True)
        if download and self.on_colab:
            try:
                from google.colab import files
                files.download(str(target))
            except Exception as error:
                print('O download automático não abriu. O ZIP está pronto; baixe pelo painel Arquivos:',
                      target,'|',type(error).__name__,flush=True)
        return target

    def failure_export(self):
        try:
            return self.export(download=False)
        except Exception as error:
            print('Os logs permanecem em',self.storage/'logs',
                  '| a exportação não concluiu:',type(error).__name__,flush=True)
            return None

    def registration(self, public_url=''):
        if getattr(self,'on_colab',False) and not self.storage_persistent:
            print('Pré-registro ainda não declarado: conecte o Drive e gere o pacote final antes de registrar.',flush=True)
            return
        folder=self.storage/'protocol'
        path=folder/'external_registration.json'
        if path.exists():
            print('Registro já declarado:',path)
            return
        if not public_url.strip():
            print('Protocolo pronto. Publique o pacote e informe PUBLIC_REGISTRATION_URL antes de treinar.')
            return
        from urllib.parse import urlparse
        url=public_url.strip()
        parsed=urlparse(url)
        if parsed.scheme not in ('http','https') or not parsed.netloc:
            raise ValueError('Informe a URL pública real do pré-registro.')
        if any((self.storage/'runs').rglob('manifest.json')):
            raise RuntimeError('Já existem treinos neste estudo. Um registro novo não pode ser retroativo.')
        value=dict(public_url=url,registered_before_first_training=True,
                   protocol_sha256=hashlib.sha256((folder/'locked_protocol.json').read_bytes()).hexdigest(),
                   model_revisions_sha256=hashlib.sha256((folder/'model_revisions.json').read_bytes()).hexdigest(),
                   note='Declaração do pesquisador. A cronologia exige auditoria do registro externo.')
        temp=path.with_suffix('.partial')
        temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
        temp.replace(path)
        print('URL declarada. A data pública exige verificação independente.')

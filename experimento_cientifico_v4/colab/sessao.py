"""Sessão dos notebooks v4: ambiente isolado, execução registrada e exportação.

O kernel do notebook não importa os módulos científicos. Cada etapa roda em um
subprocesso com PYTHONPATH fixado no checkout do commit congelado; as etapas
de GPU usam um ambiente virtual próprio que enxerga o PyTorch/CUDA do Colab
(--system-site-packages) e recebe somente as versões fixadas das bibliotecas
de treino. Assim o PyTorch do Colab não é reinstalado nem alterado.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import deque
from pathlib import Path

GPU_STAGES = {'train', 'evaluate', 'monitor', 'controls'}
PERSISTENT_STAGES = {'train', 'evaluate', 'monitor', 'controls', 'blind', 'register'}
MIN_LOCAL_FREE_GIB = 20.0
MIN_STORAGE_FREE_GIB = 3.0
MIN_VRAM_GIB = 14.0


class StageBlocked(RuntimeError):
    """Etapa impedida por um critério do protocolo ou por falta de recurso."""


REASONS = {
    'gate_not_run': 'a triagem DEV ainda não foi importada; execute o notebook 00 com o mesmo STUDY_ID',
    'not_prepared': 'o protocolo ainda não foi preparado; execute o notebook 00 com o mesmo STUDY_ID',
    'external_registration_missing': 'informe PUBLIC_REGISTRATION_URL no notebook 00 depois de publicar o registro',
    'registration_invalid_or_hash_mismatch': 'o registro declarado não corresponde aos hashes do protocolo',
}


def explain(reasons):
    """Traduz os códigos de bloqueio para instruções objetivas."""
    out = []
    for code in reasons:
        parts = code.split('_')
        if '_below_' in code:
            limit = parts[-1].replace('.', ',')
            out.append(f'{parts[0].upper()} {parts[1]} abaixo de {limit} na triagem DEV (critério do protocolo)')
        elif code.endswith('_missing_or_altered'):
            out.append(f'arquivo da triagem {parts[0].upper()} {parts[1]} ausente ou alterado')
        elif code.endswith('_incomplete'):
            out.append(f'triagem {parts[0].upper()} {parts[1]} incompleta')
        else:
            out.append(REASONS.get(code, code))
    return '; '.join(out)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_logged(command, log_path, env=None, cwd=None):
    """Mostra a saída em tempo real, grava o log e inclui o final do log no erro."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=40)
    started = time.time()
    with log_path.open('w', encoding='utf-8') as log:
        log.write('$ ' + ' '.join(map(str, command)) + '\n')
        with subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1,
                              env=env, cwd=cwd) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
                tail.append(line)
            code = process.wait()
        log.write(f'\n[exit={code} seconds={time.time() - started:.1f}]\n')
    if code:
        raise RuntimeError(f'A etapa terminou com código {code}. Log completo: {log_path}\n' + ''.join(tail))
    return code


def create_virtualenv(target, root, logs, force_fallback=False):
    """Cria o ambiente com --system-site-packages.

    O Python do Colab não inclui ensurepip; nesse caso `python -m venv` falha e o
    virtualenv é instalado em uma pasta própria (--target), sem tocar no Python
    do sistema, e usado para criar o ambiente.
    """
    target, root, logs = Path(target), Path(root), Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    if not force_fallback:
        result = subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(target)],
                                capture_output=True, text=True)
        (logs / 'venv.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode == 0 and (target / 'bin' / 'pip').exists():
            return 'venv'
        shutil.rmtree(target, ignore_errors=True)
        print('venv indisponível neste Python (sem ensurepip); usando virtualenv isolado.', flush=True)
    bootstrap = root / 'virtualenv-bootstrap'
    if not (bootstrap / 'virtualenv').is_dir():
        run_logged([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                    '--target', bootstrap, 'virtualenv==20.35.3'], logs / 'virtualenv_install.log')
    env = dict(os.environ, PYTHONPATH=str(bootstrap))
    run_logged([sys.executable, '-m', 'virtualenv', '--system-site-packages', target],
               logs / 'virtualenv_create.log', env=env)
    return 'virtualenv'


class Session:
    def __init__(self, repo, storage_state, code_commit, code_hashes, on_colab=False, technical_test=False):
        self.repo = Path(repo).resolve()
        self.package = self.repo / 'experimento_cientifico_v4'
        for name, expected in code_hashes.items():
            if sha256(self.package / 'cotv4' / name) != expected:
                raise RuntimeError(f'cotv4/{name} diverge do hash do notebook. Reabra o notebook pelo link do GitHub.')
        protocol = self.package / 'protocolo_v4.json'
        self.state = storage_state
        self.storage = Path(storage_state.path)
        self.code_commit = code_commit
        self.on_colab = on_colab
        self.technical_test = technical_test
        self.gpu_python = None
        self.base_env = dict(os.environ, COT_SOURCE_COMMIT=code_commit, PYTHONUNBUFFERED='1',
                             PYTHONPATH=str(self.package), TOKENIZERS_PARALLELISM='false',
                             HF_HUB_DISABLE_TELEMETRY='1', USE_TF='0', USE_FLAX='0', MPLBACKEND='Agg',
                             COT_TECHNICAL_TEST='1' if technical_test else '0')
        self.protocol_sha = sha256(protocol)

    # ------------------------------------------------------------ recursos
    def check_space(self, gpu):
        local = shutil.disk_usage('/content' if Path('/content').is_dir() else Path.home()).free / 1024 ** 3
        storage = shutil.disk_usage(self.storage).free / 1024 ** 3
        problems = []
        if gpu and local < MIN_LOCAL_FREE_GIB:
            problems.append(f'Disco local com {local:.1f} GiB livres; são necessários {MIN_LOCAL_FREE_GIB:.0f} GiB '
                            'para modelos e ambiente. Libere espaço ou reinicie o ambiente.')
        if storage < MIN_STORAGE_FREE_GIB:
            problems.append(f'Armazenamento do estudo com {storage:.1f} GiB livres; são necessários '
                            f'{MIN_STORAGE_FREE_GIB:.0f} GiB para checkpoints e resultados.')
        if problems:
            raise StageBlocked('\n'.join(problems))
        return dict(local_free_gib=round(local, 1), storage_free_gib=round(storage, 1))

    def gpu_environment(self):
        if self.gpu_python is not None:
            return self.gpu_python
        probe = subprocess.run([sys.executable, '-c', (
            'import json,torch;ok=torch.cuda.is_available();'
            'print(json.dumps(dict(ok=ok,torch=torch.__version__,cuda=torch.version.cuda,'
            'name=torch.cuda.get_device_name(0) if ok else None,'
            'vram=torch.cuda.get_device_properties(0).total_memory/1024**3 if ok else 0)))')],
            capture_output=True, text=True)
        if probe.returncode:
            raise StageBlocked('PyTorch não pôde ser importado neste ambiente:\n' + probe.stderr[-2000:])
        info = json.loads(probe.stdout.strip().splitlines()[-1])
        if not info['ok']:
            raise StageBlocked('GPU indisponível. No Colab: Ambiente de execução > Alterar tipo de ambiente '
                               '> GPU (T4 ou superior) e execute novamente.')
        if info['vram'] < MIN_VRAM_GIB:
            raise StageBlocked(f'GPU {info["name"]} com {info["vram"]:.1f} GiB; o protocolo requer '
                               f'pelo menos {MIN_VRAM_GIB:.0f} GiB (T4 ou superior).')
        requirements = self.package / 'requirements-gpu.txt'
        signature = hashlib.sha256(json.dumps(dict(req=requirements.read_text(), py=sys.version,
                                                   torch=info['torch'], cuda=info['cuda'])).encode()).hexdigest()[:16]
        root = Path(os.environ.get('COT_ENV_ROOT', '/content/cot-v4-envs' if Path('/content').is_dir()
                                   else Path.home() / '.cache' / 'cot-v4-envs'))
        target = root / ('gpu-' + signature)
        python = target / 'bin' / 'python'
        ready = target / 'cot-ready.json'
        logs = self.storage / 'logs' / 'ambiente'
        if not ready.is_file():
            if target.exists():
                shutil.rmtree(target)
            print('Preparando ambiente isolado (uma vez por sessão). O PyTorch do Colab é preservado.', flush=True)
            create_virtualenv(target, root, logs)
            run_logged([python, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                        '-r', requirements], logs / 'pip_install.log')
            run_logged([python, self.package / 'colab' / 'verificar_gpu.py', str(MIN_VRAM_GIB)],
                       logs / 'verificar_gpu.log', env=self.base_env)
            freeze = subprocess.run([python, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout
            (logs / 'pip_freeze.txt').write_text(freeze, encoding='utf-8')
            ready.write_text(json.dumps(dict(info, signature=signature), indent=2), encoding='utf-8')
        self.gpu_python = python
        print(f'GPU: {info["name"]} | {info["vram"]:.1f} GiB | PyTorch {info["torch"]} | CUDA {info["cuda"]}',
              flush=True)
        return python

    # ------------------------------------------------------------ execução
    def execute(self, stage, mode='sft', max_jobs=1, device=None, url=''):
        gpu = stage in GPU_STAGES and device != 'cpu'
        if stage in PERSISTENT_STAGES and self.on_colab and not self.state.persistent:
            if not self.technical_test:
                raise StageBlocked(f'Etapa "{stage}" aguardando armazenamento persistente (Drive). '
                                   'Os arquivos locais seriam perdidos ao fim da sessão.')
            print(f'TESTE TÉCNICO em pasta temporária: "{stage}" será perdido ao fim da sessão '
                  'e não constitui evidência científica.', flush=True)
        self.check_space(gpu)
        python = self.gpu_environment() if gpu else sys.executable
        operation = self.storage / 'operations' / f'{stage}_{mode}.json'
        operation.parent.mkdir(parents=True, exist_ok=True)
        if operation.exists():
            operation.unlink()
        args = [python, '-u', '-m', 'cotv4.runtime', stage, '--storage', str(self.storage), '--mode', mode,
                '--max-jobs', str(max_jobs), '--device', 'cuda' if gpu else 'cpu', '--result-json', str(operation),
                '--quiet']
        if url:
            args += ['--url', url]
        try:
            run_logged(args, self.storage / 'logs' / f'{stage}_{mode}.log', env=self.base_env, cwd=self.package)
        except RuntimeError:
            self.export(download=False, label='diagnostico_falha')
            raise
        return json.loads(operation.read_text(encoding='utf-8'))

    def status(self):
        """Estado de cada etapa a partir dos arquivos do estudo (não de mensagens)."""
        rows = []
        protocol = self.storage / 'protocol' / 'locked_protocol.json'
        rows.append(('00 Protocolo e dados', 'CONCLUÍDA' if protocol.exists() else 'NÃO EXECUTADA', ''))
        gate_path = self.storage / 'analysis' / 'gate' / 'manifest.json'
        result = None
        if protocol.exists():
            try:
                result = self.execute('status')
            except (RuntimeError, StageBlocked) as error:
                rows.append(('Estado do estudo', 'ERRO', str(error).splitlines()[0]))
        if result:
            gate = result['gate']
            rows.append(('Triagem DEV (65% por nível)', {'passed': 'CONCLUÍDA', 'failed': 'BLOQUEADA'}.get(
                gate['status'], 'NÃO EXECUTADA'), ', '.join(gate.get('reasons', [])) or
                ' '.join(f'{k}={v["accuracy"]:.1%}' for k, v in gate.get('results', {}).items())))
            reg = result['registration']
            rows.append(('Registro público', 'CONCLUÍDA' if reg['passed'] else
                         ('NÃO NECESSÁRIO (teste técnico)' if self.technical_test else 'PENDENTE'), reg['reason']))
            for mode, label in (('sft', '01 Treinos SFT'), ('grpo', '02 Treinos GRPO')):
                done, total = result['training'][mode], result['expected'][mode]
                state = 'CONCLUÍDA' if done == total else ('BLOQUEADA' if not gate['passed'] and not self.technical_test
                                                           else 'EM ANDAMENTO' if done else 'NÃO EXECUTADA')
                rows.append((label, state, f'{done}/{total}'))
            done = sum(result['evaluation'].values())
            total = sum(result['expected'].values())
            rows.append(('03 Avaliação TEST', 'CONCLUÍDA' if done == total else
                         'BLOQUEADA' if not result['all_training_complete'] else 'EM ANDAMENTO', f'{done}/{total}'))
        elif not gate_path.exists():
            rows.append(('Triagem DEV', 'NÃO EXECUTADA', ''))
        verdict = self.storage / 'analysis' / 'final_verdict.json'
        if verdict.exists():
            data = json.loads(verdict.read_text(encoding='utf-8'))
            rows.append(('04 Relatório', data['verdict'], ', '.join(data['blocked_by'])[:120]))
        width = max(len(r[0]) for r in rows)
        print('\nSITUAÇÃO DO ESTUDO' + (' — TESTE TÉCNICO (não é evidência científica)' if self.technical_test else ''))
        for name, state, detail in rows:
            print(f'  {name.ljust(width)}  {state:<16} {detail}')
        print('  Armazenamento:', self.storage, '| persistente:', self.state.persistent, flush=True)
        return rows

    # ------------------------------------------------------------ exportação
    def _write_zip(self, temporary, folders, manifest):
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            for folder in folders:
                root = self.storage / folder
                if not root.exists():
                    continue
                for path in sorted(root.rglob('*')):
                    parts = set(path.relative_to(self.storage).parts)
                    if (path.is_file() and not parts & {'adapter', 'checkpoints'} and
                            path.suffix in ('.json', '.jsonl', '.csv', '.md', '.log', '.txt')):
                        relative = str(path.relative_to(self.storage))
                        archive.write(path, relative)
                        manifest[relative] = sha256(path)
            for extra in ('PROTOCOLO_v4.md', 'protocolo_v4.json'):
                archive.write(self.package / extra, extra)
                manifest[extra] = sha256(self.package / extra)
            archive.writestr('MANIFESTO_SHA256.json', json.dumps(
                dict(code_commit=self.code_commit, protocol_sha256=self.protocol_sha,
                     technical_test=self.technical_test, files=manifest), indent=2, ensure_ascii=False))

    def export(self, registration=False, download=True, label=None):
        name = label or ('pacote_registro_v4' if registration else 'resultados_v4')
        if self.technical_test:
            name += '_TESTE_TECNICO'
        target = self.storage / 'downloads' / f'{name}.zip'
        target.parent.mkdir(parents=True, exist_ok=True)
        folders = (('protocol', 'data', 'analysis/gate') if registration else
                   ('protocol', 'operations', 'logs', 'analysis', 'human', 'runs'))
        temporary = target.with_name(target.name + '.partial')
        manifest = {}
        try:
            self._write_zip(temporary, folders, manifest)
        except BaseException:
            temporary.unlink(missing_ok=True)  # nunca deixa um ZIP parcial com nome final
            raise
        temporary.replace(target)
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('ZIP corrompido: ' + str(target))
        print(f'ZIP salvo: {target} ({target.stat().st_size / 1024:.0f} KiB, SHA256 {sha256(target)[:16]}…)', flush=True)
        if not self.state.persistent and self.on_colab:
            print('Atenção: o ZIP está em pasta temporária; baixe-o antes de encerrar a sessão.', flush=True)
        if download and self.on_colab:
            try:
                from google.colab import files
                files.download(str(target))
            except Exception as error:
                print(f'O download automático não abriu ({type(error).__name__}). O arquivo continua em {target}.\n'
                      'Baixe pelo painel Arquivos (ícone de pasta à esquerda): localize o caminho acima, '
                      'clique nos três pontos ao lado do arquivo e escolha "Fazer download".', flush=True)
        return target

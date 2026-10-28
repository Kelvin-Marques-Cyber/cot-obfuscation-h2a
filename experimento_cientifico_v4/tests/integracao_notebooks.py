"""Executa todas as células de código dos cinco notebooks v4 com o Colab simulado.

Cenários: (A) `drive.mount` falha sempre com `ValueError: mount failed`;
(B) falha transitória seguida de sucesso; (C) Drive já montado, com arquivos
do estudo preservados e reexecução sem duplicação; (D) sessão reutilizada em
que um notebook de outra revisão carrega seus próprios módulos. O download do
navegador falha em todos os cenários. A autenticação real do Google Drive não
é exercitada: ela depende da conta e da interface do Colab.

Uso: python tests/integracao_notebooks.py [--repo-dir CHECKOUT] [--report JSON]
Requer rede apenas para conferir as revisões no Hugging Face Hub (opcional).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import shutil
import time
import types
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
CREATED = []


def workdir(prefix):
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=os.environ.get('COT_TEST_TMP')))
    CREATED.append(path)
    return path


NOTEBOOKS = ['00_protocolo_dados_e_triagem.ipynb', '01_treinamento_sft.ipynb', '02_treinamento_grpo.ipynb',
             '03_avaliacao_test.ipynb', '04_monitor_humanos_e_inferencia.ipynb']


def fake_colab(mount):
    drive = types.SimpleNamespace(mount=Mock(side_effect=mount))
    files = types.SimpleNamespace(download=Mock(side_effect=RuntimeError('frontend unavailable')))
    colab = types.ModuleType('google.colab')
    colab.drive, colab.files = drive, files
    google = types.ModuleType('google')
    google.__path__ = []
    google.colab = colab
    return google, colab, drive, files


def run_notebook(path, namespace, overrides=None):
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        for index, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = cell['source']
            for old, new in (overrides or {}).items():
                source = source.replace(old, new)
            exec(compile(source, f'{Path(path).name}:cell-{index}', 'exec'), namespace)
    return output.getvalue()


def scenario_permanent_failure(directory, repo_dir):
    google, colab, drive, files = fake_colab(ValueError('mount failed'))
    temp = workdir('cotv4-A-')
    env = {'COT_LOCAL_ROOT': str(temp / 'content'), 'COT_DRIVE_MOUNTPOINT': str(temp / 'drive'),
           'COT_REPO_DIR': str(repo_dir), 'COT_ENV_ROOT': str(temp / 'envs')}
    outputs = {}
    with patch.dict(sys.modules, {'google': google, 'google.colab': colab}), patch.dict(os.environ, env):
        os.environ.pop('COT_STORAGE_ROOT', None)
        storage = None
        for name in NOTEBOOKS:
            namespace = {'__name__': '__main__'}
            outputs[name] = run_notebook(directory / name, namespace)
            assert namespace['ON_COLAB'] is True
            assert namespace['STATE'].persistent is False
            assert namespace['STUDY'].gpu_python is None, 'nenhum ambiente GPU pode ser criado sem Drive'
            storage = storage or namespace['STATE'].path
            assert namespace['STATE'].path == storage
    assert drive.mount.call_count == 10, drive.mount.call_count
    assert not (temp / 'drive').exists(), 'pasta MyDrive falsa não pode ser criada'
    assert not (storage / 'runs').exists()
    for zip_name in ('pacote_registro_v4.zip', 'resultados_v4.zip'):
        assert zipfile.is_zipfile(storage / 'downloads' / zip_name), zip_name
    assert files.download.call_count == 2
    assert 'painel Arquivos' in outputs[NOTEBOOKS[0]]
    assert 'MODO LOCAL TEMPORÁRIO' in outputs[NOTEBOOKS[0]]
    for name in NOTEBOOKS[1:3]:
        assert 'ETAPA BLOQUEADA' in outputs[name], name
    verdict = json.loads((storage / 'analysis' / 'final_verdict.json').read_text())
    assert verdict['verdict'] == 'INSUFFICIENT_EVIDENCE'
    return dict(mount_errors=drive.mount.call_count, download_errors=files.download.call_count,
                storage=str(storage), blocked_messages={n: outputs[n].count('ETAPA BLOQUEADA') for n in NOTEBOOKS})


def scenario_transient_then_existing(directory, repo_dir):
    temp = workdir('cotv4-BC-')
    mountpoint = temp / 'drive'
    mounted = [False]

    def mount(path, timeout_ms=None):
        if not hasattr(mount, 'calls'):
            mount.calls = 0
        mount.calls += 1
        if mount.calls == 1:
            raise ValueError('mount failed')
        (mountpoint / 'MyDrive').mkdir(parents=True, exist_ok=True)
        mounted[0] = True
    google, colab, drive, files = fake_colab(mount)
    env = {'COT_LOCAL_ROOT': str(temp / 'content'), 'COT_DRIVE_MOUNTPOINT': str(mountpoint),
           'COT_REPO_DIR': str(repo_dir), 'COT_ENV_ROOT': str(temp / 'envs')}
    real_ismount = os.path.ismount
    fake_ismount = lambda p: (mounted[0] and Path(p) == mountpoint) or real_ismount(p)
    with patch.dict(sys.modules, {'google': google, 'google.colab': colab}), patch.dict(os.environ, env), \
            patch('os.path.ismount', side_effect=fake_ismount):
        os.environ.pop('COT_STORAGE_ROOT', None)
        namespace = {'__name__': '__main__'}
        first = run_notebook(directory / NOTEBOOKS[0], namespace)
        state = namespace['STATE']
        assert state.persistent and state.backend == 'drive', state.as_dict()
        assert drive.mount.call_count == 2
        protocol = state.path / 'protocol' / 'locked_protocol.json'
        data = state.path / 'data' / 'sft.jsonl'
        marker = state.path / 'human' / 'nota_do_pesquisador.txt'
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('arquivo existente que deve ser preservado')
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (protocol, data, marker)}
        files_before = sorted(str(p.relative_to(state.path)) for p in state.path.rglob('*')
                              if p.is_file() and 'downloads' not in p.parts and 'logs' not in p.parts
                              and 'operations' not in p.parts)
        # (C) Drive já montado: nenhuma nova montagem e nenhum arquivo duplicado.
        namespace = {'__name__': '__main__'}
        second = run_notebook(directory / NOTEBOOKS[0], namespace)
        assert drive.mount.call_count == 2, 'montagem existente deve ser reutilizada'
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
        assert before == after
        files_after = sorted(str(p.relative_to(state.path)) for p in state.path.rglob('*')
                             if p.is_file() and 'downloads' not in p.parts and 'logs' not in p.parts
                             and 'operations' not in p.parts)
        assert files_before == files_after, set(files_after) ^ set(files_before)
        assert 'Armazenamento persistente verificado' in second
    return dict(transient_mount_calls=2, reused_mount=True, files_preserved=len(before),
                files_unchanged_after_rerun=len(files_after))


def scenario_reused_session(directory, repo_dir):
    """Dois notebooks com commits distintos no mesmo namespace Python."""
    google, colab, drive, files = fake_colab(ValueError('mount failed'))
    temp = workdir('cotv4-D-')
    env = {'COT_LOCAL_ROOT': str(temp / 'content'), 'COT_DRIVE_MOUNTPOINT': str(temp / 'drive'),
           'COT_REPO_DIR': str(repo_dir)}
    namespace = {'__name__': '__main__'}
    with patch.dict(sys.modules, {'google': google, 'google.colab': colab}), patch.dict(os.environ, env):
        source = json.loads((directory / NOTEBOOKS[0]).read_text())['cells'][1]['source']
        commit = source.split('CODE_COMMIT = "')[1][:40]
        other = 'e' * 40
        run_notebook(directory / NOTEBOOKS[0], namespace)
        first_module, first_study = namespace['sessao'], namespace['STUDY']
        locked = first_study.storage / 'protocol' / 'locked_protocol.json'
        before = locked.read_bytes()
        try:
            run_notebook(directory / NOTEBOOKS[0], namespace, {commit: other})
        except RuntimeError as error:
            assert 'Protocolo congelado diverge' in str(error)
        else:
            raise AssertionError('outro commit no mesmo STUDY_ID deveria ser recusado')
        assert locked.read_bytes() == before
        run_notebook(directory / NOTEBOOKS[0], namespace,
                     {commit: other, '"estudo-prospectivo-2026-v4"': '"estudo-revisao-alternativa"'})
        second_module, second_study = namespace['sessao'], namespace['STUDY']
    assert first_module is not second_module
    assert first_module.__name__ == f'cotv4_sessao_{commit[:12]}'
    assert second_module.__name__ == f'cotv4_sessao_{other[:12]}'
    assert type(second_study).__module__ == second_module.__name__
    assert second_study.code_commit == other and first_study.code_commit == commit
    assert not any(name == 'cotv4' or name.startswith('cotv4.') for name in sys.modules), \
        'o kernel não deve importar o pacote científico'
    return dict(distinct_modules=[first_module.__name__, second_module.__name__],
                other_commit_same_study_refused=True,
                kernel_imports_scientific_package=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-dir', default=str(ROOT.parent))
    parser.add_argument('--notebooks', default=str(ROOT))
    parser.add_argument('--report')
    args = parser.parse_args()
    directory, repo = Path(args.notebooks), Path(args.repo_dir)
    started = time.time()
    try:
        report = run_all(directory, repo)
    finally:
        for path in CREATED:
            shutil.rmtree(path, ignore_errors=True)
    report['seconds'] = round(time.time() - started, 1)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        Path(args.report).write_text(text + '\n', encoding='utf-8')
    print(text)


def run_all(directory, repo):
    return dict(permanent_mount_failure=scenario_permanent_failure(directory, repo),
                transient_failure_and_existing_drive=scenario_transient_then_existing(directory, repo),
                reused_session=scenario_reused_session(directory, repo), training_started=False,
                scientific_results_generated=False, live_colab_authentication_tested=False)


if __name__ == '__main__':
    main()

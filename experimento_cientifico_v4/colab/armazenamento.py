"""Armazenamento do Colab com montagem verificada do Google Drive.

Princípios: no máximo duas tentativas de montagem; o Drive só é considerado
disponível quando o ponto de montagem é real e um arquivo de teste é gravado e
relido; nenhuma pasta local chamada MyDrive é criada para simular
persistência; arquivos existentes nunca são apagados ou movidos.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

DRIVE_SUBDIR = 'MyDrive/CoT_Obfuscation_Lab/prospectivo_v4'
RECONNECT = ('Para reconectar: mantenha STORAGE_MODE="auto", execute de novo a primeira célula '
             'e autorize a conta. Se o erro persistir, use Ambiente de execução > Desconectar e '
             'excluir ambiente de execução e abra o notebook novamente pelo link do GitHub.')


class StorageState:
    def __init__(self, path, backend, persistent, diagnostics, free_gib):
        self.path = Path(path)
        self.backend = backend
        self.persistent = persistent
        self.diagnostics = diagnostics
        self.free_gib = free_gib

    def as_dict(self):
        return dict(path=str(self.path), backend=self.backend, persistent=self.persistent,
                    diagnostics=self.diagnostics, free_gib=self.free_gib)


def writable_directory(path):
    """Grava, relê e remove um arquivo de teste no diretório pretendido."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix='.cot-probe-', delete=True) as probe:
            probe.write(b'cot-storage-probe')
            probe.flush()
            os.fsync(probe.fileno())
            probe.seek(0)
            return probe.read() == b'cot-storage-probe'
    except OSError:
        return False


def drive_available(mountpoint):
    mountpoint = Path(mountpoint)
    return os.path.ismount(mountpoint) and (mountpoint / 'MyDrive').is_dir()


def free_gib(path):
    try:
        return round(shutil.disk_usage(path).free / 1024 ** 3, 2)
    except OSError:
        return None


def _stale_mountpoint(mountpoint):
    mountpoint = Path(mountpoint)
    return mountpoint.is_dir() and not os.path.ismount(mountpoint) and any(mountpoint.iterdir())


def prepare_storage(local_root, study_id, *, on_colab=False, mode='auto', storage_override=None,
                    drive_module=None, mountpoint='/content/drive', attempts=2, timeout_ms=120000):
    if mode not in ('auto', 'local'):
        raise ValueError('STORAGE_MODE deve ser "auto" ou "local".')
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    mountpoint = Path(mountpoint)
    diagnostics = []
    parent, backend, persistent = None, 'local', not on_colab
    if storage_override:
        candidate = Path(storage_override).expanduser().resolve()
        inside_drive = on_colab and candidate.is_relative_to(mountpoint.resolve())
        if inside_drive and not drive_available(mountpoint):
            diagnostics.append('O caminho indicado está no Drive, mas o Drive não está montado.')
        elif writable_directory(candidate):
            parent, backend = candidate, 'drive' if inside_drive else 'custom'
            persistent = True if inside_drive else not on_colab
        else:
            diagnostics.append('O caminho indicado não aceita gravação.')
    elif on_colab and mode == 'auto' and drive_module is not None:
        candidate = mountpoint / DRIVE_SUBDIR
        for attempt in range(1, max(1, min(int(attempts), 2)) + 1):
            if drive_available(mountpoint):
                if writable_directory(candidate):
                    parent, backend, persistent = candidate, 'drive', True
                    break
                diagnostics.append(f'Tentativa {attempt}: Drive montado, mas sem gravação em {candidate}.')
            if _stale_mountpoint(mountpoint):
                diagnostics.append(f'{mountpoint} contém arquivos sem estar montado; a montagem não '
                                   'será forçada para não apagar nada.')
                break
            try:
                print(f'Conectando o Google Drive (tentativa {attempt}/2). Autorize a conta.', flush=True)
                drive_module.mount(str(mountpoint), timeout_ms=timeout_ms)
            except Exception as error:  # a API do Colab sinaliza falhas como ValueError/TimeoutError
                diagnostics.append(f'Tentativa {attempt}: {type(error).__name__}: {error}')
                print(f'A montagem falhou ({type(error).__name__}: {error}).', flush=True)
            if drive_available(mountpoint) and writable_directory(candidate):
                parent, backend, persistent = candidate, 'drive', True
                break
        if parent is None:
            diagnostics.append('Drive indisponível após as tentativas e o teste de gravação.')
    elif on_colab and mode == 'local':
        diagnostics.append('STORAGE_MODE="local": o Drive não foi solicitado.')
    if parent is None:
        parent = local_root / 'studies'
    storage = parent / study_id
    if not writable_directory(storage):
        diagnostics.append(f'Sem gravação em {storage}; usando pasta local temporária.')
        storage, backend, persistent = local_root / 'studies' / study_id, 'local', not on_colab
        if not writable_directory(storage):
            raise OSError('Nem a pasta local aceita gravação. Verifique o espaço em disco da sessão '
                          f'(livre: {free_gib(local_root)} GiB).')
    state = StorageState(storage, backend, persistent, diagnostics, free_gib(storage))
    log = local_root / f'armazenamento-{study_id}.json'
    log.write_text(json.dumps(state.as_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
    if persistent:
        print(f'Armazenamento persistente verificado: {storage} (livre: {state.free_gib} GiB)', flush=True)
    else:
        print('MODO LOCAL TEMPORÁRIO:', storage, flush=True)
        print('Os arquivos desta pasta são apagados quando a sessão do Colab termina.', flush=True)
        print('O notebook 00 pode preparar e exportar o pacote; treinos e avaliações ficam pausados.', flush=True)
        for line in diagnostics:
            print('  -', line, flush=True)
        print(RECONNECT, flush=True)
    return state

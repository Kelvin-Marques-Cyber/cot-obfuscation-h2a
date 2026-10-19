"""Recoverable Colab storage setup, independent of the frozen scientific code."""
from pathlib import Path
import json
import os
import tempfile


class StorageState:
    def __init__(self, path, backend, persistent, diagnostics):
        self.path=Path(path)
        self.backend=backend
        self.persistent=persistent
        self.diagnostics=diagnostics


def writable_directory(path):
    """Probe the intended directory; a mount returning successfully is insufficient."""
    path=Path(path)
    try:
        path.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=path) as probe:
            probe.write(b'cot-storage-probe')
            probe.flush()
            probe.seek(0)
            if probe.read()!=b'cot-storage-probe':
                return False
        return True
    except OSError:
        return False


def drive_available(mountpoint):
    mountpoint=Path(mountpoint)
    # A stale MyDrive directory is not evidence of a mounted Google Drive.
    return os.path.ismount(mountpoint) and (mountpoint/'MyDrive').is_dir()


def prepare_storage(local_root, study_id, *, on_colab=False, mode='auto',
                    storage_override=None, drive_module=None,
                    mountpoint='/content/drive', attempts=2, timeout_ms=60000):
    if mode not in ('auto','local'):
        raise ValueError('STORAGE_MODE deve ser auto ou local.')
    local_root=Path(local_root)
    local_root.mkdir(parents=True,exist_ok=True)
    mountpoint=Path(mountpoint)
    diagnostics=[]
    parent=None
    backend='local'
    persistent=not on_colab
    if storage_override:
        candidate=Path(storage_override).expanduser().resolve()
        if on_colab and candidate.is_relative_to(mountpoint.resolve()):
            if drive_available(mountpoint) and writable_directory(candidate):
                parent=candidate
                backend='drive'
                persistent=True
            else:
                diagnostics.append('O diretório solicitado no Drive não está montado ou gravável.')
        elif writable_directory(candidate):
            parent=candidate
            backend='custom'
        else:
            diagnostics.append('O diretório solicitado não está gravável; usando a pasta local.')
    elif on_colab and mode=='auto' and drive_module is not None:
        candidate=mountpoint/'MyDrive/CoT_Obfuscation_Lab/confirmatory_v3'
        for attempt in range(max(1,min(int(attempts),2))):
            if drive_available(mountpoint) and writable_directory(candidate):
                parent=candidate
                backend='drive'
                persistent=True
                break
            try:
                print(f'Conectando o Drive (tentativa {attempt+1}/2). Autorize a conta quando solicitado.',flush=True)
                drive_module.mount(str(mountpoint),timeout_ms=timeout_ms)
            except Exception as error:
                diagnostics.append(f'Tentativa {attempt+1}: {type(error).__name__}: {error}')
                print('O Drive não conectou. '+('Tentando novamente.' if attempt==0 else 'Usando preparação local.'),flush=True)
            # Check even when mount raised: a partially completed connection can recover.
            if drive_available(mountpoint) and writable_directory(candidate):
                parent=candidate
                backend='drive'
                persistent=True
                break
        if parent is None:
            diagnostics.append('Drive indisponível após as tentativas de conexão e gravação.')
    if parent is None:
        parent=local_root/'studies'
    storage=parent/study_id
    if not writable_directory(storage):
        # Never create files below an unmounted Drive to pretend persistence.
        storage=local_root/'studies'/study_id
        backend='local'
        persistent=not on_colab
        diagnostics.append('Falha ao gravar no destino; usando preparação local.')
        if not writable_directory(storage):
            raise OSError('A pasta local também não aceita gravação. Verifique o espaço da sessão.')
    state=StorageState(storage,backend,persistent,diagnostics)
    info=dict(backend=backend,persistent=persistent,path=str(storage),diagnostics=diagnostics)
    log=local_root/f'storage-{study_id}.json'
    log.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    if persistent:
        print('Armazenamento verificado:',storage,flush=True)
    else:
        print('MODO LOCAL TEMPORÁRIO:',storage,flush=True)
        print('O notebook 00 pode preparar e baixar o protocolo. Os treinos aguardam o Drive.',flush=True)
        print('Os arquivos locais são perdidos ao encerrar a máquina. Gere o pacote final no Drive antes de registrar o estudo.',flush=True)
        print('Para tentar conectar novamente, deixe STORAGE_MODE="auto" e reexecute a primeira célula.',flush=True)
    return state

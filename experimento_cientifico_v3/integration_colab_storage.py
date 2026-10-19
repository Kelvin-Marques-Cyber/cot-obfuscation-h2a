"""Execute all five notebooks with a failing Colab Drive and download frontend.

Requires huggingface_hub and network access for model-revision metadata only.
No model weights, GPU, training, or scientific results are generated.
Optional COT_REPO_DIR and COT_SETUP_REPO_DIR reuse exact pinned checkouts.
"""
import json
import os
from pathlib import Path
import tempfile
import types
from unittest.mock import Mock,patch
import zipfile


def main():
    root=Path(__file__).resolve().parent
    drive=types.SimpleNamespace(mount=Mock(side_effect=ValueError('mount failed')))
    files=types.SimpleNamespace(download=Mock(side_effect=RuntimeError('frontend unavailable')))
    google=types.ModuleType('google')
    google.__path__=[]
    colab=types.ModuleType('google.colab')
    colab.drive=drive
    colab.files=files
    google.colab=colab
    completed=[]
    with tempfile.TemporaryDirectory() as temp, \
         patch.dict('sys.modules',{'google':google,'google.colab':colab}), \
         patch.dict(os.environ,{'COT_LOCAL_ROOT':temp,
                               'COT_DRIVE_MOUNTPOINT':str(Path(temp)/'unmounted-drive')}):
        os.environ.pop('COT_STORAGE_ROOT',None)
        storage_path=None
        for path in sorted(root.glob('*.ipynb')):
            nb=json.loads(path.read_text(encoding='utf-8'))
            namespace={'__name__':'__main__'}
            for i,cell in enumerate(nb['cells']):
                if cell['cell_type']=='code':
                    exec(compile(cell['source'],f'{path.name}:cell-{i}','exec'),namespace)
            assert namespace['ON_COLAB'] is True
            assert namespace['state'].persistent is False
            assert namespace['STUDY'].gpu_session is None
            assert namespace['SOURCE_COMMIT']=='fde856f895c841274b93b34f4675642508af102f'
            if storage_path is not None:
                assert namespace['STORAGE']==storage_path
            storage_path=namespace['STORAGE']
            assert not (storage_path/'runs').exists()
            assert not (storage_path/'protocol/external_registration.json').exists()
            completed.append(path.name)
            print('MOUNT_FAILED_ALL_CELLS_OK',path.name,flush=True)
        assert len(completed)==5
        assert drive.mount.call_count==10
        assert files.download.call_count==2
        for name in ('pacote_para_pre_registro.zip','resultados_cientificos_v3.zip'):
            assert zipfile.is_zipfile(storage_path/'downloads'/name)
        report=json.loads((storage_path/'analysis/final_verdict.json').read_text())
        assert report['verdict']=='INSUFFICIENT_EVIDENCE'
        print(json.dumps(dict(status='passed',notebooks=completed,
                              mount_errors_injected=drive.mount.call_count,
                              download_errors_injected=files.download.call_count,
                              full_code_cells_executed=True,training_started=False,
                              scientific_results_generated=False)),flush=True)


if __name__=='__main__':
    main()

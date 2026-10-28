"""Injeção de falhas no armazenamento e na sessão do Colab (CPU, sem GPU).

As falhas de `drive.mount` e `files.download` são simuladas; a autenticação
real do Google Drive depende da conta e não é exercitada aqui.
"""
import hashlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'colab'))
import armazenamento  # noqa: E402
import sessao  # noqa: E402

REPO = ROOT.parent
HASHES = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'cotv4').glob('*.py')}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mount = self.root / 'drive'
        self.drive = Mock()
        self.out = io.StringIO()

    def prepare(self, **kwargs):
        with redirect_stdout(self.out):
            return armazenamento.prepare_storage(self.root / 'local', 'estudo', on_colab=True,
                                                 drive_module=self.drive, mountpoint=self.mount, **kwargs)

    def test_permanent_mount_failure_is_local_and_temporary(self):
        self.drive.mount.side_effect = ValueError('mount failed')
        state = self.prepare()
        self.assertEqual(self.drive.mount.call_count, 2)
        self.assertFalse(state.persistent)
        self.assertEqual(state.backend, 'local')
        self.assertFalse(self.mount.exists(), 'nenhuma pasta MyDrive falsa pode ser criada')
        self.assertIn('MODO LOCAL TEMPORÁRIO', self.out.getvalue())
        self.assertIn('ValueError: mount failed', ' '.join(state.diagnostics))
        self.assertIn('reconectar', self.out.getvalue())

    def test_transient_failure_then_success(self):
        mounted = [False]

        def mount(path, timeout_ms):
            if self.drive.mount.call_count == 1:
                raise ValueError('mount failed')
            (self.mount / 'MyDrive').mkdir(parents=True)
            mounted[0] = True
        self.drive.mount.side_effect = mount
        with patch.object(armazenamento.os.path, 'ismount', side_effect=lambda p: mounted[0]):
            state = self.prepare()
        self.assertTrue(state.persistent)
        self.assertEqual(state.backend, 'drive')
        self.assertEqual(self.drive.mount.call_count, 2)
        self.assertTrue(str(state.path).endswith('prospectivo_v4/estudo'))

    def test_existing_mount_is_reused_and_files_preserved(self):
        study = self.mount / armazenamento.DRIVE_SUBDIR / 'estudo'
        study.mkdir(parents=True)
        (self.mount / 'MyDrive').mkdir(exist_ok=True)
        old = study / 'protocol.json'
        old.write_bytes(b'existente')
        with patch.object(armazenamento.os.path, 'ismount', return_value=True):
            state = self.prepare()
        self.drive.mount.assert_not_called()
        self.assertTrue(state.persistent)
        self.assertEqual(old.read_bytes(), b'existente')
        self.assertEqual(sorted(p.name for p in study.iterdir()), ['protocol.json'])

    def test_stale_folder_is_not_drive_and_is_not_forced(self):
        (self.mount / 'MyDrive').mkdir(parents=True)
        (self.mount / 'MyDrive' / 'antigo.txt').write_text('preservar')
        state = self.prepare()
        self.drive.mount.assert_not_called()
        self.assertFalse(state.persistent)
        self.assertEqual((self.mount / 'MyDrive' / 'antigo.txt').read_text(), 'preservar')
        self.assertIn('sem estar montado', ' '.join(state.diagnostics))

    def test_mounted_but_not_writable_falls_back(self):
        with patch.object(armazenamento.os.path, 'ismount', return_value=True), \
             patch.object(armazenamento, 'writable_directory',
                          side_effect=lambda p: not str(p).startswith(str(self.mount))):
            (self.mount / 'MyDrive').mkdir(parents=True)
            state = self.prepare()
        self.assertFalse(state.persistent)
        self.assertIn('sem gravação', ' '.join(state.diagnostics))

    def test_local_mode_skips_mount(self):
        state = self.prepare(mode='local')
        self.drive.mount.assert_not_called()
        self.assertFalse(state.persistent)

    def test_outside_colab_local_folder_is_persistent(self):
        with redirect_stdout(self.out):
            state = armazenamento.prepare_storage(self.root / 'local', 'estudo', on_colab=False)
        self.assertTrue(state.persistent)


class VirtualenvFallbackTests(unittest.TestCase):
    """Cria de fato um ambiente pelo caminho alternativo (requer acesso ao PyPI)."""
    def test_fallback_without_ensurepip_creates_isolated_env_with_system_packages(self):
        import subprocess
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            kind = sessao.create_virtualenv(root / 'env', root, root / 'logs', force_fallback=True)
            python = root / 'env' / 'bin' / 'python'
            self.assertEqual(kind, 'virtualenv')
            probe = subprocess.run([str(python), '-c', 'import sys; print(sys.prefix != sys.base_prefix)'],
                                   capture_output=True, text=True)
            self.assertEqual(probe.stdout.strip(), 'True')
            self.assertTrue((root / 'env' / 'pyvenv.cfg').read_text().count('include-system-site-packages = true'))


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = io.StringIO()
        self.env = patch.dict(os.environ, {'COT_TECHNICAL_TEST': '0'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def session(self, persistent, on_colab=True):
        state = armazenamento.StorageState(self.root / 'estudo', 'drive' if persistent else 'local',
                                           persistent, [], 100.0)
        state.path.mkdir(parents=True, exist_ok=True)
        return sessao.Session(REPO, state, 'b' * 40, HASHES, on_colab=on_colab)

    def test_code_hash_mismatch_is_refused(self):
        state = armazenamento.StorageState(self.root, 'local', True, [], 1.0)
        with self.assertRaisesRegex(RuntimeError, 'diverge'):
            sessao.Session(REPO, state, 'b' * 40, dict(HASHES, **{'runtime.py': '0' * 64}))

    def test_persistent_stages_wait_for_drive(self):
        session = self.session(persistent=False)
        for stage in ('train', 'evaluate', 'register'):
            with self.assertRaisesRegex(sessao.StageBlocked, 'persistente'):
                session.execute(stage)

    def test_technical_test_may_use_temporary_storage_with_warning(self):
        state = armazenamento.StorageState(self.root / 'tecnico', 'local', False, [], 100.0)
        state.path.mkdir(parents=True)
        session = sessao.Session(REPO, state, 'b' * 40, HASHES, on_colab=True, technical_test=True)
        with redirect_stdout(self.out), patch.object(session, 'check_space', side_effect=sessao.StageBlocked('x')):
            with self.assertRaisesRegex(sessao.StageBlocked, 'x'):
                session.execute('train')
        self.assertIn('TESTE TÉCNICO em pasta temporária', self.out.getvalue())

    def test_real_subprocess_prepare_status_and_export_with_download_failure(self):
        session = self.session(persistent=False)
        with redirect_stdout(self.out):
            prepared = session.execute('prepare')
            self.assertEqual(prepared['datasets']['sft']['test_per_level'], {'A': 220, 'C': 220})
            again = session.execute('prepare')
            self.assertEqual(prepared, again)
            status = session.execute('status')
            self.assertTrue(status['prepared'])
            self.assertEqual(status['expected'], {'sft': 120, 'grpo': 120})
            runs = session.storage / 'runs' / 'sft' / 'x'
            (runs / 'adapter').mkdir(parents=True)
            (runs / 'adapter' / 'adapter_config.json').write_text('{}')
            (runs / 'checkpoints').mkdir()
            (runs / 'checkpoints' / 'trainer_state.json').write_text('{}')
            (runs / 'manifest.json').write_text('{}')
            files = types.SimpleNamespace(download=Mock(side_effect=RuntimeError('frontend unavailable')))
            colab = types.ModuleType('google.colab')
            colab.files = files
            google = types.ModuleType('google')
            google.colab = colab
            with patch.dict(sys.modules, {'google': google, 'google.colab': colab}):
                target = session.export(registration=True)
                results = session.export()
        self.assertEqual(files.download.call_count, 2)
        self.assertIn('painel Arquivos', self.out.getvalue())
        self.assertIn('pasta temporária', self.out.getvalue())
        for zip_path in (target, results):
            self.assertTrue(zipfile.is_zipfile(zip_path))
        with zipfile.ZipFile(results) as archive:
            names = archive.namelist()
        self.assertIn('runs/sft/x/manifest.json', names)
        self.assertFalse(any('adapter' in n or 'checkpoints' in n for n in names))
        with zipfile.ZipFile(target) as archive:
            manifest = json.loads(archive.read('MANIFESTO_SHA256.json'))
        self.assertIn('data/sft.jsonl', manifest['files'])

    def test_failed_stage_reports_log_and_exports_diagnostic(self):
        session = self.session(persistent=True, on_colab=False)
        with redirect_stdout(self.out):
            with self.assertRaisesRegex(RuntimeError, 'Log completo'):
                session.execute('evaluate', device='cpu')  # sem protocolo preparado
        self.assertTrue((session.storage / 'downloads' / 'diagnostico_falha.zip').is_file())
        self.assertTrue((session.storage / 'logs' / 'evaluate_sft.log').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)

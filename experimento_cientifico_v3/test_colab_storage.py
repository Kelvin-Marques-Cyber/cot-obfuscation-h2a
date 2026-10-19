"""Failure injection for Colab mounting and temporary-storage safeguards."""
from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch
import zipfile

import colab_storage as storage
from colab_support import ScientificSession


class StorageRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.local=self.root/'local'
        self.mount=self.root/'drive'
        self.drive=Mock()
        self.output=io.StringIO()
        self.redirect=redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__,None,None,None)

    def prepare(self,**kwargs):
        return storage.prepare_storage(self.local,'study',on_colab=True,
            drive_module=self.drive,mountpoint=self.mount,**kwargs)

    def session(self,persistent=False):
        repo=Path(__file__).resolve().parents[1]
        script=repo/'experimento_cientifico_v3/scientific_runtime.py'
        return ScientificSession(repo,self.local/'study','fde856f895c841274b93b34f4675642508af102f',
            hashlib.sha256(script.read_bytes()).hexdigest(),on_colab=True,
            storage_persistent=persistent,storage_backend='drive' if persistent else 'local')

    def test_mount_failed_twice_falls_back_without_traceback_or_fake_drive(self):
        self.drive.mount.side_effect=ValueError('mount failed')
        state=self.prepare()
        self.assertEqual(self.drive.mount.call_count,2)
        self.assertFalse(state.persistent)
        self.assertEqual(state.path,self.local/'studies/study')
        self.assertFalse(self.mount.exists())
        self.assertIn('MODO LOCAL TEMPORÁRIO',self.output.getvalue())
        self.assertNotIn('Traceback',self.output.getvalue())
        for call in self.drive.mount.call_args_list:
            self.assertNotIn('force_remount',call.kwargs)

    def test_first_mount_failure_then_success_uses_drive(self):
        mounted=[False]
        def mount(*args,**kwargs):
            if self.drive.mount.call_count==1:
                raise ValueError('mount failed')
            (self.mount/'MyDrive').mkdir(parents=True)
            mounted[0]=True
        self.drive.mount.side_effect=mount
        with patch.object(storage.os.path,'ismount',side_effect=lambda p:mounted[0]):
            state=self.prepare()
        self.assertTrue(state.persistent)
        self.assertEqual(state.backend,'drive')
        self.assertEqual(self.drive.mount.call_count,2)

    def test_existing_mount_is_reused_and_study_files_are_preserved(self):
        path=self.mount/'MyDrive/CoT_Obfuscation_Lab/confirmatory_v3/study'
        path.mkdir(parents=True)
        previous=path/'checkpoint.json'
        previous.write_bytes(b'unchanged-existing-study')
        with patch.object(storage.os.path,'ismount',return_value=True):
            state=self.prepare()
        self.drive.mount.assert_not_called()
        self.assertTrue(state.persistent)
        self.assertEqual(previous.read_bytes(),b'unchanged-existing-study')

    def test_stale_mydrive_folder_is_not_treated_as_persistent(self):
        (self.mount/'MyDrive').mkdir(parents=True)
        self.drive.mount.side_effect=ValueError('mount failed')
        with patch.object(storage.os.path,'ismount',return_value=False):
            state=self.prepare()
        self.assertFalse(state.persistent)
        self.assertFalse((self.mount/'MyDrive/CoT_Obfuscation_Lab').exists())

    def test_mount_without_write_access_falls_back(self):
        (self.mount/'MyDrive').mkdir(parents=True)
        original=storage.writable_directory
        def writable(path):
            return False if Path(path).is_relative_to(self.mount) else original(path)
        with patch.object(storage.os.path,'ismount',return_value=True), \
             patch.object(storage,'writable_directory',side_effect=writable):
            state=self.prepare()
        self.assertFalse(state.persistent)
        self.assertEqual(state.backend,'local')

    def test_local_choice_skips_authorization(self):
        state=self.prepare(mode='local')
        self.drive.mount.assert_not_called()
        self.assertFalse(state.persistent)

    def test_unmounted_drive_override_does_not_create_fake_persistent_files(self):
        with patch.object(storage.os.path,'ismount',return_value=False):
            state=self.prepare(storage_override=self.mount/'MyDrive/custom')
        self.assertFalse(state.persistent)
        self.assertFalse(self.mount.exists())

    def test_user_interrupt_is_respected(self):
        self.drive.mount.side_effect=KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.prepare()
        self.assertEqual(self.drive.mount.call_count,1)

    def test_temporary_storage_never_starts_training_or_downloads_gpu_stack(self):
        session=self.session()
        session.session_type=Mock(side_effect=AssertionError('GPU setup must not start'))
        self.assertIsNone(session.gpu())
        for stage in ('dev','train','evaluate','monitor','controls'):
            self.assertEqual(session.execute(stage,gpu=True)['status'],'waiting_for_persistent_storage')
        session.session_type.assert_not_called()
        self.assertFalse((session.storage/'runs').exists())

    def test_temporary_preview_does_not_declare_preregistration(self):
        session=self.session()
        session.registration('https://osf.io/example/')
        self.assertFalse((session.storage/'protocol/external_registration.json').exists())

    def test_browser_download_failure_keeps_valid_zip_and_manual_path(self):
        session=self.session()
        (session.storage/'protocol').mkdir()
        (session.storage/'protocol/locked_protocol.json').write_text('{}')
        colab=types.ModuleType('google.colab')
        colab.files=types.SimpleNamespace(download=Mock(side_effect=RuntimeError('frontend unavailable')))
        with patch.dict('sys.modules',{'google.colab':colab}):
            target=session.export(registration=True)
        self.assertTrue(zipfile.is_zipfile(target))
        with zipfile.ZipFile(target) as archive:
            self.assertIn('protocol/locked_protocol.json',archive.namelist())
        self.assertIn('painel Arquivos',self.output.getvalue())

    def test_export_failure_does_not_replace_original_gpu_failure(self):
        session=self.session(persistent=True)
        broken=Mock()
        broken.ensure_environment.side_effect=RuntimeError('original environment failure')
        session.session_type=Mock(return_value=broken)
        with patch.object(session,'export',side_effect=OSError('export unavailable')):
            with self.assertRaisesRegex(RuntimeError,'original environment failure'):
                session.gpu()


if __name__=='__main__':
    unittest.main(verbosity=2)

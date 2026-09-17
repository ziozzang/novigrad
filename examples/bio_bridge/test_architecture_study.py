import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import architecture_study as study


class LockTests(unittest.TestCase):
    def test_changed_frozen_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'weights'
            path.write_bytes(b'original')
            with patch.object(study, 'ROOT', root):
                expected = study.hashes([path])
                study.check_hashes(expected)
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'changed frozen artifact'):
                    study.check_hashes(expected)

    def test_final_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'holdout-final.json').write_text('{}')
            with patch.object(study, 'OUT', out), patch.object(study, 'checked_lock', return_value={}):
                with self.assertRaises(FileExistsError):
                    study.final()

    def test_corrupted_case_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'holdout').mkdir()
            features = out / 'holdout/holdout-embeddings.safetensors'
            features.write_bytes(b'not inspected until provenance passes')
            (out / 'holdout/provenance.json').write_text(json.dumps({
                'cases_sha256': 'wrong', 'embedding_sha256': study.sha(features)}))
            with patch.object(study, 'OUT', out):
                with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                    study.input_set('holdout')


if __name__ == '__main__':
    unittest.main()

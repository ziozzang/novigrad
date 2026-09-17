import hashlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from precise_pipeline import check_hashes,compare_tensors
from encode_precise_holdout import validate_cases,DEFAULT_CASES
import json

class PipelineTests(unittest.TestCase):
    def test_hash_lock_detects_mutation_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'artifact';p.write_bytes(b'original')
            manifest={'artifact':hashlib.sha256(b'original').hexdigest()}
            check_hashes(manifest,root)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError):check_hashes(manifest,root)
            p.unlink()
            with self.assertRaises(ValueError):check_hashes(manifest,root)
    def test_tensor_replay_rejects_numerical_drift(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.safetensors',Path(d)/'b.safetensors'
            save_file({'x':np.ones(2,np.float32)},str(a));save_file({'x':np.ones(2,np.float32)},str(b))
            self.assertTrue(compare_tensors(a,b))
            save_file({'x':np.zeros(2,np.float32)},str(b))
            with self.assertRaises(AssertionError):compare_tensors(a,b)
    def test_holdout_rejects_broken_pair(self):
        rows=json.loads(DEFAULT_CASES.read_text());validate_cases(rows)
        rows[0]['language']='ko' if rows[0]['language']=='en' else 'en'
        with self.assertRaises(ValueError):validate_cases(rows)

if __name__=='__main__':unittest.main()

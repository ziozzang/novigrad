import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_neural_bridge as runner

class RunnerTests(unittest.TestCase):
    def test_model_revision_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);metadata=root/'results/thought-bridge';metadata.mkdir(parents=True)
            model=root/'model';model.mkdir();weight=model/'model.safetensors';weight.write_bytes(b'original')
            (metadata/'description-provenance.json').write_text(json.dumps({'weight_sha256':{'model.safetensors':hashlib.sha256(b'original').hexdigest()}}))
            with patch.object(runner.bridge,'ROOT',root):
                runner.validate_model(model)
                weight.write_bytes(b'different')
                with self.assertRaises(ValueError):runner.validate_model(model)
    def test_context_restores_paths_on_failure(self):
        original=(runner.bridge.OUT,runner.pipeline.OUT)
        with self.assertRaises(RuntimeError):
            with runner.experiment_directory(runner.bridge.ROOT/'results/test-isolation'):
                raise RuntimeError('failure')
        self.assertEqual(original,(runner.bridge.OUT,runner.pipeline.OUT))
        with self.assertRaises(ValueError):
            with runner.experiment_directory('/tmp/outside-neural-run'):pass

if __name__=='__main__':unittest.main()

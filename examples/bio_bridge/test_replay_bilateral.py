import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from replay_bilateral import digest,relative_name,verify_model_directory,relocated_models


class RelocationTests(unittest.TestCase):
    def test_identical_relocated_content_and_tokenizer_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'model.safetensors').write_bytes(b'weights');(root/'tokenizer.json').write_text('{}')
            expected={p.name:digest(p) for p in root.iterdir()}
            self.assertEqual(verify_model_directory(root,expected),root.resolve())
            (root/'tokenizer.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify_model_directory(root,expected)
    def test_unexpected_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'config.json').write_text('{}')
            with self.assertRaises(ValueError):verify_model_directory(root,{})
    def test_lookup_override_restores_original_runtime(self):
        class Runtime:
            _cache={}
            @classmethod
            def load(cls,model_path='/old/model',device='mps'):
                return str(model_path),device
        original=Runtime.__dict__['load']
        sentinel=lambda models: None
        study=SimpleNamespace(check_models=sentinel)
        lm=SimpleNamespace(Runtime=Runtime,MODEL_PATH=Path('/old/model'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            with self.assertRaises(RuntimeError):
                with relocated_models(study,lm,{'embedding':root,'function':root}):
                    study.check_models({'embedding':{'files':{}},'function':{'files':{}}})
                    self.assertEqual(Runtime.load(),(str(root),'mps'))
                    raise RuntimeError('test cleanup')
        self.assertIs(Runtime.__dict__['load'],original)
        self.assertIs(study.check_models,sentinel)

    def test_relative_paths(self):
        for path in ('../weights','/tmp/weights','a/../weights','a\\weights','a//b','C:/weights'):
            with self.subTest(path=path),self.assertRaises(ValueError):relative_name(path)


if __name__=='__main__':unittest.main()

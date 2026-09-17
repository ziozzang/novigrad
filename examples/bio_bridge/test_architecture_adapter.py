import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import architecture_adapter as aa
from safetensors.torch import save_file

class ArchitectureAdapterTest(unittest.TestCase):
    def test_lora_gradient_and_zero_start_delta(self):
        torch.manual_seed(1);a=torch.randn(aa.INPUT_DIM,4,requires_grad=True);b=torch.zeros(4,aa.OUTPUT_DIM,requires_grad=True)
        self.assertEqual(torch.count_nonzero(a@b),0)
        (torch.randn(3,aa.INPUT_DIM)@(a@b)).sum().backward()
        self.assertGreater(torch.count_nonzero(b.grad),0)

    def test_pca_discards_centering_null_direction(self):
        rng=np.random.default_rng(2);x=rng.normal(size=(9,aa.INPUT_DIM)).astype(np.float32)
        _,w=aa._pca(x)
        self.assertLessEqual(np.linalg.matrix_rank(w),len(x)-1)

    def test_fractional_labels_rejected(self):
        with self.assertRaisesRegex(ValueError,'integers'):
            aa._validate_data(np.zeros((2,aa.INPUT_DIM),np.float32),np.array([0.,1.5]),'x')

    def test_checkpoint_metadata_and_schema_rejected(self):
        base={'mean':torch.zeros(aa.INPUT_DIM),'w0':torch.zeros(aa.INPUT_DIM,aa.OUTPUT_DIM),'head_weight':torch.zeros(aa.CLASSES,aa.OUTPUT_DIM),'head_bias':torch.zeros(aa.CLASSES)}
        metadata={'format':'novigrad.architecture_adapter','version':'1','kind':'frozen','seed':'3','scope':'direct_embedding_adapter_not_connectome'}
        with tempfile.TemporaryDirectory() as directory:
            wrong_meta=Path(directory)/'wrong-meta.safetensors'
            save_file(base,str(wrong_meta),metadata={**metadata,'version':'2'})
            with self.assertRaisesRegex(ValueError,'metadata'):aa.load_state(wrong_meta,'frozen',3)
            wrong_schema=Path(directory)/'wrong-schema.safetensors'
            save_file({**base,'extra':torch.zeros(1)},str(wrong_schema),metadata=metadata)
            with self.assertRaisesRegex(ValueError,'tensor names'):aa.load_state(wrong_schema,'frozen',3)
            wrong_shape=Path(directory)/'wrong-shape.safetensors'
            save_file({**base,'head_bias':torch.zeros(5)},str(wrong_shape),metadata=metadata)
            with self.assertRaisesRegex(ValueError,'head_bias'):aa.load_state(wrong_shape,'frozen',3)

    def test_train_roundtrip_no_mutation_and_overwrite_refusal(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(12,aa.INPUT_DIM)).astype(np.float32);y=np.arange(12)%aa.CLASSES
        vx=rng.normal(size=(8,aa.INPUT_DIM)).astype(np.float32);vy=np.arange(8)%aa.CLASSES;original=x.copy()
        data=lambda:{'train':(x,y,None),'validation':(vx,vy,None)}
        with tempfile.TemporaryDirectory() as directory, patch.object(aa,'dataset',data),patch.object(aa,'RANKS',(4,)),patch.object(aa,'SEEDS',(3,)),patch.object(aa,'EPOCHS',2):
            report=aa.train_all(directory)
            self.assertTrue(np.array_equal(x,original));self.assertTrue(report['models']['rank4'][0]['cpu_roundtrip_exact'])
            self.assertGreater(report['models']['full'][0]['parameters'],report['models']['rank4'][0]['parameters'])
            self.assertLessEqual(report['models']['rank4'][0]['effective_linear_rank'],aa.CLASSES)
            result=aa.evaluate_all(directory,vx,vy)
            self.assertEqual(set(result),{'frozen','rank4','full'});self.assertEqual(len(result['rank4'][0]['predictions']),len(vx))
            with self.assertRaises(FileExistsError):aa.train_all(directory)

if __name__=='__main__':unittest.main()

import unittest
import tempfile
import numpy as np
from safetensors.numpy import save_file,load_file
from thought_embedding import unit,fit_alignment,decode,retrieve


class SemanticDecoderTests(unittest.TestCase):
    def test_alignment_recovers_simple_mapping_and_zero(self):
        x=np.eye(4);basis,coef=fit_alignment(x,x)
        np.testing.assert_allclose(decode(basis,coef,x),x,atol=1e-12)
        np.testing.assert_array_equal(decode(basis,coef,np.zeros((1,4))),np.zeros((1,4)))

    def test_candidates_report_similarity_not_confidence(self):
        descriptions=[{'id':str(i),'text':str(i),'category':str(i)} for i in range(4)]
        result=retrieve(np.eye(4),descriptions,np.eye(4))
        self.assertEqual([r['category'] for r in result],list('0123'))
        self.assertNotIn('confidence',result[0])
        empty=retrieve(np.zeros((1,4)),descriptions,np.eye(4))[0]
        self.assertIsNone(empty['category'])
        self.assertEqual(empty['top3'],[])

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):unit([[float('nan')]])
        with self.assertRaises(ValueError):fit_alignment(np.eye(2),np.eye(3))
        with self.assertRaises(ValueError):fit_alignment(np.eye(2),np.eye(2),ridge=0)

    def test_fortran_features_have_exact_checkpoint_roundtrip(self):
        hidden=np.asfortranarray(np.arange(24,dtype=float).reshape(4,6)+1)
        basis,coef=fit_alignment(hidden,np.eye(4))
        self.assertTrue(basis.flags.c_contiguous)
        with tempfile.NamedTemporaryFile(suffix='.safetensors') as file:
            save_file({'basis':basis,'coef':coef},file.name)
            saved=load_file(file.name)
            np.testing.assert_array_equal(saved['basis'],basis)
            np.testing.assert_array_equal(decode(basis,coef,hidden),decode(saved['basis'],saved['coef'],hidden))


if __name__=='__main__':unittest.main()

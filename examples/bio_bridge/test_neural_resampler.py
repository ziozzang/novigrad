import tempfile,unittest
from pathlib import Path
import torch
from neural_resampler import NeuralResampler,save_resampler,load_resampler

class NeuralResamplerTest(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(4);x=torch.randn(2,5,32);mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
        sites=torch.tensor([[0,1,2,3,0],[3,2,1,0,0]]);times=torch.tensor([[0.,1.,2.,3.,4.],[1.,2.,3.,0.,0.]])
        return x,mask,sites,times,torch.tensor([4.,3.])

    def test_shapes_gradients_and_parameter_match(self):
        x,mask,sites,times,obs=self.inputs();models=[NeuralResampler(m) for m in ('learned','fixed','pooled_mlp')]
        for model in models:
            value=model(x,mask,sites,times,obs);self.assertEqual(value.shape,(2,4,32));value.square().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual([m.trainable_parameters for m in models],[4608,4480,4667])
        difference=abs(models[2].trainable_parameters-models[0].trainable_parameters)/models[0].trainable_parameters
        self.assertLess(difference,.05)
        self.assertFalse(models[1].queries.requires_grad)
        self.assertIsNotNone(models[1].attention.in_proj_weight.grad)

    def test_padding_and_joint_permutation_invariance(self):
        x,mask,sites,times,obs=self.inputs();model=NeuralResampler('learned').eval();base=model(x,mask,sites,times,obs)
        permutation=torch.tensor([3,0,4,1,2]);permuted=model(x[:,permutation],mask[:,permutation],sites[:,permutation],times[:,permutation],obs)
        self.assertTrue(torch.allclose(base,permuted,atol=1e-6))
        extra_x=torch.cat((x,torch.randn(2,3,32)*100),1);extra_mask=torch.cat((mask,torch.zeros(2,3,dtype=torch.bool)),1)
        extra_sites=torch.cat((sites,torch.full((2,3),99)),1);extra_times=torch.cat((times,torch.full((2,3),999.)),1)
        padded=model(extra_x,extra_mask,extra_sites,extra_times,obs)
        self.assertTrue(torch.allclose(base,padded,atol=1e-6))

    def test_clock_origin_and_metadata_behavior(self):
        x,mask,sites,times,obs=self.inputs();model=NeuralResampler('learned').eval()
        base=model(x,mask,sites,times,obs)
        shifted=model(x,mask,sites,times+1234.,obs+1234.)
        self.assertTrue(torch.allclose(base,shifted,atol=1e-6))
        changed_sites=sites.clone();changed_sites[mask]=(changed_sites[mask]+1)%4
        self.assertFalse(torch.equal(base,model(x,mask,changed_sites,times,obs)))
        changed_times=times.clone();changed_times[mask]-=.25
        self.assertFalse(torch.equal(base,model(x,mask,sites,changed_times,obs)))
        # Arbitrarily large padded metadata remains fully ignored.
        padded_times=times.clone();padded_times[~mask]=torch.finfo(times.dtype).max
        padded_sites=sites.clone();padded_sites[~mask]=torch.iinfo(sites.dtype).max
        self.assertTrue(torch.equal(base,model(x,mask,padded_sites,padded_times,obs)))

    def test_validation_failures(self):
        x,mask,sites,times,obs=self.inputs();model=NeuralResampler()
        bad=times.clone();bad[0,0]=5
        with self.assertRaisesRegex(ValueError,'future'):model(x,mask,sites,bad,obs)
        with self.assertRaisesRegex(ValueError,'bool'):model(x,mask.int(),sites,times,obs)
        empty=mask.clone();empty[0]=False
        with self.assertRaisesRegex(ValueError,'at least one'):model(x,empty,sites,times,obs)
        fractional=sites.float()
        with self.assertRaisesRegex(ValueError,'integer'):model(x,mask,fractional,times,obs)
        nonfinite=x.clone();nonfinite[0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite'):model(nonfinite,mask,sites,times,obs)

    def test_safetensors_exact_roundtrip(self):
        x,mask,sites,times,obs=self.inputs()
        for mode in ('learned','fixed','pooled_mlp'):
            model=NeuralResampler(mode).eval();before=model(x,mask,sites,times,obs)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.safetensors';save_resampler(model,path);restored=load_resampler(path).eval();after=restored(x,mask,sites,times,obs)
            self.assertTrue(torch.equal(before,after))

if __name__=='__main__':unittest.main()

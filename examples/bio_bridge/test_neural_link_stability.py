import tempfile,unittest
from pathlib import Path
import numpy as np
from safetensors.numpy import save_file
from neural_link_stability import (AffineCalibrator,RecordingBatch,align_paired,
 apply_recording,fit_recordings,load_recording,save_recording,verify_recording)

class StabilityTests(unittest.TestCase):
 def test_affine_roundtrip_and_ridge(self):
  x=np.arange(24,dtype=float).reshape(6,4);y=2*x+3;c=AffineCalibrator.fit(x,y,.37,np.arange(4))
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'c.safetensors';c.save(p);restored=AffineCalibrator.load(p)
   self.assertEqual(restored.ridge,.37);np.testing.assert_array_equal(c.infer(x),restored.infer(x))
 def test_recording_formats(self):
  b=RecordingBatch(np.arange(3),np.arange(3.),np.eye(3,dtype=np.float32),np.array([10,20,30]),np.arange(3))
  with tempfile.TemporaryDirectory() as d:
   for suffix in ('.npz','.safetensors'):
    p=Path(d)/('r'+suffix);save_recording(p,b);r=load_recording(p,3,b.channel_ids);np.testing.assert_array_equal(r.neural_features,b.neural_features)
 def test_end_to_end_aligns_shuffled_ids(self):
  ids=np.array([10,20,30,40]);channels=np.array([101,103]);clean=np.array([[1.,2.],[2.,4.],[3.,6.],[4.,8.]],np.float32)
  observed=clean*2+1; order=np.array([2,0,3,1]); times=np.array([.1,.2,.3,.4])
  ref=RecordingBatch(ids,times,clean,channels,np.array([0,1,0,1]));obs=RecordingBatch(ids[order],times[order],observed[order],channels,np.array([0,0,1,1]))
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);op=d/'obs.npz';rp=d/'ref.safetensors';cp=d/'cal.safetensors';ip=d/'input.npz';out=d/'out.safetensors';manifest=d/'apply.json'
   save_recording(op,obs);save_recording(rp,ref);fit_recordings(op,rp,cp,ridge=1e-8)
   save_recording(ip,obs);apply_recording(cp,ip,out,manifest);status=verify_recording(cp,ip,out,manifest)
   self.assertTrue(status['verified'])
   bad=__import__('json').loads(manifest.read_text());bad['operation']='wrong';manifest.write_text(__import__('json').dumps(bad))
   with self.assertRaises(ValueError):verify_recording(cp,ip,out,manifest)
   corrected=load_recording(out);self.assertLess(np.max(np.abs(corrected.neural_features-clean[order])),1e-5)
 def test_rejects_corruption_and_pair_mismatch(self):
  ch=np.array([1,2]);x=np.eye(2)
  with self.assertRaises(ValueError):RecordingBatch(np.array([1.,2.]),np.arange(2.),x,ch).validate()
  with self.assertRaises(ValueError):RecordingBatch(np.array([1,1]),np.arange(2.),x,ch).validate()
  with self.assertRaises(ValueError):RecordingBatch(np.arange(2),np.arange(2.),np.array([[1.,np.nan],[0,1]]),ch).validate()
  with self.assertRaises(ValueError):RecordingBatch(np.arange(2),np.arange(2.),x,ch,np.array([0.,1.])).validate()
  a=RecordingBatch(np.array([1,2]),np.array([0.,1.]),x,ch)
  with self.assertRaises(ValueError):align_paired(a,RecordingBatch(np.array([1,3]),np.array([0.,1.]),x,ch))
  with self.assertRaises(ValueError):align_paired(a,RecordingBatch(np.array([1,2]),np.array([0.,1.]),x,ch[::-1]))
 def test_safetensor_metadata_required(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'bad.safetensors';save_file({'sample_ids':np.arange(2,dtype=np.int64),'time':np.arange(2.),'neural_features':np.eye(2),'channel_ids':np.arange(2,dtype=np.int64)},str(p),metadata={'format':'wrong'})
   with self.assertRaises(ValueError):load_recording(p)
   c=Path(d)/'bad-cal.safetensors';save_file({'anchors':np.ones((2,3)),'dual':np.ones((2,2))},str(c),metadata={'format':'novi.paired-affine-ridge','version':'2','ridge':'.1','input_features':'2','output_features':'2'})
   with self.assertRaises(ValueError):AffineCalibrator.load(c)
if __name__=='__main__':unittest.main()

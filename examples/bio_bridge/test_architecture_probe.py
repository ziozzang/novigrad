import tempfile,unittest
from pathlib import Path
import numpy as np
from architecture_probe import (ALTERNATE_RULES,CYCLIC_RULES,ARCHITECTURES,
 design,evaluate,evaluate_saved,expand,fit_probes,run_probe,save_bundle,schedule_controls,validate_rules)
from precise_bridge import dataset

class ArchitectureProbeTests(unittest.TestCase):
 def setUp(self):
  rng=np.random.default_rng(4);self.labels=np.tile(np.arange(4),3);self.reps={'h':rng.normal(size=(12,20)),'z':rng.normal(size=(12,5)),'embedding':rng.normal(size=(12,7))}
 def test_rule_tables_are_balanced_permutations(self):
  for table in (CYCLIC_RULES,ALTERNATE_RULES):np.testing.assert_array_equal(validate_rules(table),table)
  with self.assertRaises(ValueError):validate_rules(np.zeros((4,4),dtype=int))
 def test_expansion_and_missing_interaction(self):
  x,c,y,row=expand(self.reps['h'],self.labels,CYCLIC_RULES)
  self.assertEqual(x.shape,(48,20));self.assertEqual(set(c),set(range(4)))
  X=design('interaction_kc',{'h':x,'z':self.reps['z'][row],'embedding':self.reps['embedding'][row]},np.full(48,-1))
  np.testing.assert_array_equal(X[:,:-1],0);np.testing.assert_array_equal(X[:,-1],1)
 def test_capacity_match_and_all_models(self):
  probes,details=fit_probes(self.reps,self.labels,CYCLIC_RULES)
  self.assertEqual(set(probes),set(ARCHITECTURES));add=details['additive_kc']['coefficient_count'];matched=details['matched_interaction_kc']['coefficient_count']
  self.assertLessEqual(abs(add-matched),4*4)
  for p in probes.values():
   result=evaluate(p,self.reps,self.labels,CYCLIC_RULES);self.assertEqual(len(result['predictions']),48);np.testing.assert_allclose(np.sum(result['probabilities'],axis=1),1)
 def test_host_latch_is_exact_external_replay(self):
  probes,_=fit_probes(self.reps,self.labels,CYCLIC_RULES)
  for probe in probes.values():self.assertTrue(schedule_controls(probe,self.reps,self.labels,CYCLIC_RULES)['repeated_vs_cue_once_latch_exact'])

 def test_saved_bundle_logits_exact_and_tamper_rejected(self):
  data=dataset();train_x,train_y,_=data["train"];val_x,val_y,_=data["validation"]
  with tempfile.TemporaryDirectory() as d:
   save_bundle(d);saved=evaluate_saved(d,val_x,val_y);direct=run_probe(train_x,train_y,val_x,val_y)
   for kind in ARCHITECTURES:
    self.assertEqual(saved["cyclic"]["models"][kind]["evaluation"]["probabilities"],direct["models"][kind]["evaluation"]["probabilities"])
   weights=Path(d)/"architecture-probes.safetensors";weights.write_bytes(weights.read_bytes()+b"tamper")
   with self.assertRaises(ValueError):evaluate_saved(d,val_x,val_y)
 def test_float_labels_rejected(self):
  with self.assertRaises(ValueError):expand(self.reps['h'],self.labels.astype(float),CYCLIC_RULES)
if __name__=='__main__':unittest.main()

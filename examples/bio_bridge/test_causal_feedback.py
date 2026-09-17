import unittest
import numpy as np
import torch
from bilateral_lm import target_text
from causal_feedback import next_prefix,normalized_prior,rollout
from feedback_environment import FeedbackConfig

def raw(action):return target_text(action)
def fixtures():
 initial=torch.zeros((4,3));initial[0,0]=1.;prototypes=torch.zeros((4,4,3))
 for i in range(4):prototypes[i,i,0]=1.
 prior=np.array([.7,.1,.1,.1]);return initial,prototypes,prior

class CausalFeedbackTests(unittest.TestCase):
 def test_normalized_prior_and_decision_do_not_read_target_before_feedback(self):
  p=normalized_prior([[0.,-2.,-3.,-4.]], [2,2,2,2])[0];self.assertAlmostEqual(p.sum(),1.)
  initial,prototypes,prior=fixtures();generate=lambda prefix,key,index:raw(0)
  a=rollout(initial,prototypes,prior,0,[0,1,2,3],'belief_full',FeedbackConfig(),5,generate)
  b=rollout(initial,prototypes,prior,1,[0,1,2,3],'belief_full',FeedbackConfig(),5,generate)
  self.assertEqual(a['trace'][0]['command'],b['trace'][0]['command']);self.assertEqual(a['trace'][0]['belief'],b['trace'][0]['belief']);self.assertNotEqual(a['trace'][0]['physical_success'],b['trace'][0]['physical_success'])
 def test_fake_generation_changes_only_after_feedback_prefix(self):
  initial,prototypes,prior=fixtures();seen=[]
  def generate(prefix,key,index):
   seen.append(prefix.clone());return raw(0 if torch.equal(prefix,initial) else 1)
  result=rollout(initial,prototypes,prior,1,[0,1,2,3],'belief_full',FeedbackConfig(),0,generate)
  self.assertEqual([x['command'] for x in result['trace']],[0,1]);self.assertTrue(result['rescued']);self.assertFalse(torch.equal(seen[0],seen[1]));self.assertNotEqual(result['trace'][0]['prefix_key'],result['trace'][1]['prefix_key'])
 def test_native_mapping_updates_executed_action_not_command(self):
  initial,prototypes,prior=fixtures();result=rollout(initial,prototypes,np.full(4,.25),2,[1,0,2,3],'belief_full',FeedbackConfig(),0,lambda *args:raw(0));first=result['trace'][0]
  self.assertEqual(first['command'],0);self.assertEqual(first['executed_action'],1);self.assertEqual(first['observation']['feedback_action'],1);self.assertLess(first['next_belief'][1],first['next_belief'][0])
 def test_delayed_feedback_retains_original_action_credit(self):
  initial,prototypes,prior=fixtures();calls=iter((0,1,2));result=rollout(initial,prototypes,np.full(4,.25),2,[0,1,2,3],'belief_full',FeedbackConfig(delay=1),0,lambda *args:raw(next(calls)))
  first,second=result['trace'][:2];self.assertIsNone(first['observation']['feedback_action']);self.assertEqual(second['observation']['feedback_action'],0);self.assertLess(second['next_belief'][0],second['next_belief'][1]);self.assertEqual(second['executed_action'],1)
 def test_no_feedback_keeps_history_belief_constant(self):
  initial,prototypes,prior=fixtures();calls=iter((0,1,2));result=rollout(initial,prototypes,prior,2,[0,1,2,3],'static_prior_prefix',FeedbackConfig(),0,lambda *args:raw(next(calls)))
  for row in result['trace']:np.testing.assert_allclose(row['next_belief'],prior)
 def test_flipped_delivered_reward_does_not_change_physical_success(self):
  initial,prototypes,prior=fixtures();result=rollout(initial,prototypes,prior,0,[0,1,2,3],'flipped_feedback',FeedbackConfig(),0,lambda *args:raw(0));row=result['trace'][0]
  self.assertTrue(row['physical_success']);self.assertTrue(result['success']);self.assertEqual(row['observation']['reward'],.95);self.assertEqual(row['delivered_observation']['reward'],-.05)
  noisy=rollout(initial,prototypes,prior,0,[0,1,2,3],'belief_full',FeedbackConfig(reward_flip_probability=1.),0,lambda *args:raw(0));self.assertTrue(noisy['success']);self.assertEqual(noisy['trace'][0]['observation']['reward'],-.05)
 def test_prefix_norm_gain_and_permutation_controls(self):
  initial,prototypes,prior=fixtures();belief=np.array([0.,1.,0.,0.]);base=float(initial.norm())
  half=next_prefix(initial,prototypes,belief,'belief_half',0);full=next_prefix(initial,prototypes,belief,'belief_full',0);permuted=next_prefix(initial,prototypes,belief,'permuted_prototypes',0);selfwrite=next_prefix(initial,prototypes,belief,'self_writeback',2)
  for value in (half,full,permuted,selfwrite):self.assertAlmostEqual(float(value.norm()),base,places=6)
  self.assertFalse(torch.equal(half,full));self.assertGreater(full[1,0],0);self.assertGreater(permuted[2,0],0);self.assertGreater(selfwrite[2,0],0)
 def test_invalid_generation_ends_without_physical_action(self):
  initial,prototypes,prior=fixtures();result=rollout(initial,prototypes,prior,0,[0,1,2,3],'belief_full',FeedbackConfig(),0,lambda *args:'not a call')
  self.assertTrue(result['invalid']);self.assertEqual(result['physical_actions'],0);self.assertFalse(result['success']);self.assertEqual(len(result['trace']),1);self.assertNotIn('executed_action',result['trace'][0])
if __name__=='__main__':unittest.main()

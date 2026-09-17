import unittest
import numpy as np
from feedback_environment import FeedbackConfig,HiddenGoalEnvironment,bayes_action,feedback_events,posterior
class FeedbackEnvironmentTests(unittest.TestCase):
 def test_observation_never_contains_goal_or_latent_state(self):
  env=HiddenGoalEnvironment([2]);obs=env.reset();self.assertEqual(set(obs),{'executed_action','feedback_action','reward','done','step'});self.assertFalse(any('goal' in k or 'state' in k or 'success' in k for k in obs));obs=env.step(2);self.assertEqual(set(obs),{'executed_action','feedback_action','reward','done','step'})
 def test_reward_cost_terminal_and_budget(self):
  env=HiddenGoalEnvironment([2,3],config=FeedbackConfig(max_steps=2));env.reset();a=env.step(0);self.assertEqual(a['reward'],-.05);self.assertFalse(a['done']);b=env.step(1);self.assertTrue(b['done']);self.assertEqual(b['reward'],-.05);env.reset();c=env.step(3);self.assertTrue(c['done']);self.assertEqual(c['reward'],.95)
 def test_delay_retains_action_credit(self):
  env=HiddenGoalEnvironment([2],config=FeedbackConfig(delay=1));env.reset();first=env.step(0);self.assertIsNone(first['feedback_action']);self.assertIsNone(first['reward']);second=env.step(2);self.assertEqual(second['executed_action'],2);self.assertEqual(second['feedback_action'],0);self.assertEqual(second['reward'],-.05);self.assertTrue(second['done'])
  self.assertEqual(list(feedback_events([first,second])),[(0,-.05,2)])
 def test_permutation_maps_intended_to_executed(self):
  env=HiddenGoalEnvironment([2],action_permutation=[2,0,3,1]);env.reset();obs=env.execute(0);self.assertEqual(obs['executed_action'],2);self.assertTrue(obs['done'])
 def test_noise_is_presampled_and_shared(self):
  config=FeedbackConfig(reward_flip_probability=.1);a=HiddenGoalEnvironment([3,3],seed=7,config=config);b=HiddenGoalEnvironment([3,3],seed=7,config=config);self.assertEqual(a.noise_schedule_sha256,b.noise_schedule_sha256)
  for _ in range(2):
   a.reset();b.reset();self.assertEqual(a.step(0),b.step(0))
   while a._active:self.assertEqual(a.step(1),b.step(1))
 def test_reset_and_validation(self):
  with self.assertRaises(ValueError):HiddenGoalEnvironment([0.])
  with self.assertRaises(ValueError):HiddenGoalEnvironment([0],action_permutation=[0,0,1,2])
  env=HiddenGoalEnvironment([0]);
  with self.assertRaises(RuntimeError):env.step(0)
  env.reset()
  with self.assertRaises(RuntimeError):env.reset()
  with self.assertRaises(ValueError):env.step(True)
 def test_bayes_elimination_delay_and_no_clipping(self):
  prior=np.full(4,.25);none={'executed_action':0,'feedback_action':None,'reward':None,'done':False,'step':1};negative={'executed_action':1,'feedback_action':0,'reward':-.05,'done':False,'step':2};p=posterior(prior,[none,negative],reliability=1.);self.assertEqual(p[0],0);np.testing.assert_allclose(p[1:],1/3);self.assertEqual(bayes_action(prior,[negative],1.),1)
  positive={'executed_action':2,'feedback_action':2,'reward':.95,'done':True,'step':1};np.testing.assert_array_equal(posterior(prior,[positive],1.),[0,0,1,0])
 def test_noisy_likelihood_and_impossible_history(self):
  prior=np.full(4,.25);positive={'executed_action':0,'feedback_action':0,'reward':.95,'done':False,'step':1};p=posterior(prior,[positive],.9);self.assertGreater(p[0],p[1]);zero=np.array([1.,0,0,0]);other={**positive,'feedback_action':1}
  with self.assertRaises(ValueError):posterior(zero,[other],1.)
if __name__=='__main__':unittest.main()

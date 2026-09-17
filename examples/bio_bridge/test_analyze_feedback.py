import unittest
import numpy as np
from analyze_feedback import condition_summary,paired_cluster_bootstrap,prior_diagnostics,reachable_ceiling
class AnalyzeFeedbackTests(unittest.TestCase):
 def cases(self):return [{'class':'water' if i<2 else 'food','language':'en' if i%2==0 else 'ko','family':f'f{i//2}','pair_id':f'p{i//2}'} for i in range(4)]
 def rows(self):return [{'success':x,'physical_actions':a,'decisions':d,'invalid':inv} for x,a,d,inv in [(1,2,2,False),(0,0,1,True),(1,1,1,False),(0,3,3,False)]]
 def test_role_aligned_group_summary_and_success_conditional_decisions(self):
  s=condition_summary(self.rows(),self.cases());self.assertEqual(s['overall']['successes'],2);self.assertEqual(s['overall']['mean_physical_actions'],1.5);self.assertEqual(s['overall']['conditional_mean_decisions_among_successes'],1.5);self.assertEqual(set(s['by_language']),{'en','ko'})
 def test_pair_bootstrap_alignment_and_determinism(self):
  ids=['a','a','b','b'];x=paired_cluster_bootstrap([1,0,1,1],[0,0,1,0],ids,1000,4);y=paired_cluster_bootstrap([1,0,1,1],[0,0,1,0],ids,1000,4);self.assertEqual(x,y);self.assertEqual(x['paired_success_difference_left_minus_right'],.5)
  with self.assertRaises(ValueError):paired_cluster_bootstrap([1,0],[0,1],['a','b'])
 def test_reachability_is_actuator_combinatorics(self):
  a=reachable_ceiling({'actions':[0,3,3,3]});self.assertEqual(a['reachable_goal_count'],2);self.assertEqual(a['balanced_four_class_case_ceiling'],.5);self.assertEqual(reachable_ceiling({'actions':[0,1,2,3]})['balanced_four_class_case_ceiling'],1.)
 def test_prior_diagnostic_uses_external_labels_only_for_scoring(self):
  p=np.array([[.7,.1,.1,.1],[.1,.7,.1,.1]]);r=prior_diagnostics(p,[0,1]);self.assertEqual(r['top1_accuracy'],1.);self.assertIn('not claimed',r['boundary'])
if __name__=='__main__':unittest.main()

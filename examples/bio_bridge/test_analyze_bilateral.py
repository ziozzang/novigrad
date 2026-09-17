import unittest
import numpy as np
from analyze_bilateral import paired_cluster_bootstrap,summarize,transitions
class AnalyzeTests(unittest.TestCase):
 def cases(self):return [{'id':str(i),'pair_id':f'p{i//2}','family':f'f{i//4}','language':'en' if i%2==0 else 'ko','class':'water' if i<4 else 'food'} for i in range(8)]
 def rows(self,correct):return [{'correct':bool(x),'valid':True} for x in correct]
 def test_group_summaries(self):
  s=summarize(self.rows([1,0,1,0,1,1,0,0]),self.cases());self.assertEqual(s['overall']['correct'],4);self.assertEqual(set(s['by_language']),{'en','ko'});self.assertEqual(len(s['by_family']),2)
 def test_paired_bootstrap_is_deterministic(self):
  a=np.array([1,1,0,0,1,0,1,0]);b=np.array([0,1,0,0,0,0,1,1]);clusters=np.array([0,0,1,1,2,2,3,3]);x=paired_cluster_bootstrap(a,b,clusters,1000,9);y=paired_cluster_bootstrap(a,b,clusters,1000,9);self.assertEqual(x,y);self.assertAlmostEqual(x['paired_accuracy_difference_left_minus_right'],.125)
 def test_transitions(self):
  t=transitions([0,1,1,0],[1,0,1,0],self.cases()[:4]);self.assertEqual((t['help'],t['hurt'],t['both_correct'],t['both_wrong']),(1,1,1,1));self.assertEqual(len(t['cases']),4)
 def test_bad_bootstrap_rejected(self):
  with self.assertRaises(ValueError):paired_cluster_bootstrap([1],[0],['only'])
if __name__=='__main__':unittest.main()

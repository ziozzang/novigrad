import unittest
import numpy as np
from embedding_scenarios import margin_predictions,recognition_metrics,prototype_scores

class EmbeddingTests(unittest.TestCase):
    def test_ambiguous_margin_abstains(self):
        p=margin_predictions(np.array([[.5,.5,.1,0],[.8,.3,.2,.1]]),.1)
        self.assertEqual(p.tolist(),[-1,0])
        self.assertEqual(recognition_metrics(p,np.array([-1,0]))['balanced_accuracy'],1.)
    def test_dimensions_and_scale_normalization(self):
        x=np.eye(4); labels=np.arange(4)
        np.testing.assert_allclose(prototype_scores(x,labels,x*3,4),np.eye(4))

if __name__=='__main__':unittest.main()

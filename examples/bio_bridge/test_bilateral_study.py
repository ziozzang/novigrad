import copy
from pathlib import Path
import tempfile
import unittest
from bilateral_study import check_models, sha, compare_lm


class StudyTests(unittest.TestCase):
    def test_model_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text('{"original":true}')
            models = {'test': {'path': tmp, 'files': {'config.json': sha(path)}}}
            check_models(models)
            path.write_text('{"original":false}')
            with self.assertRaisesRegex(ValueError, 'model changed'):
                check_models(models)

    def records(self):
        row = {'index': 0, 'target': 0, 'rank_prediction': 0, 'candidate_loglikelihoods': [-1., -2., -3., -4.],
               'free_output': 'call', 'parsed_goal': 'water', 'valid': True, 'correct': True}
        return {**{k: {'rows': [copy.deepcopy(row)], 'seconds': 1.} for k in
                   ('actual', 'zero_prefix', 'row_shuffled_prefix', 'training_mean_prefix')}, 'shuffle_order': [0]}

    def test_replay_ignores_timing_but_not_semantics(self):
        a = self.records(); b = copy.deepcopy(a); b['actual']['seconds'] = 900.
        self.assertTrue(compare_lm(a, b)['checked_free_generation_exact'])
        b['actual']['rows'][0]['free_output'] = 'different'
        with self.assertRaisesRegex(AssertionError, 'generation replay'):
            compare_lm(a, b)

    def test_replay_numeric_tolerance(self):
        a = self.records(); b = copy.deepcopy(a)
        b['actual']['rows'][0]['candidate_loglikelihoods'][0] += .1
        with self.assertRaisesRegex(AssertionError, 'likelihood replay'):
            compare_lm(a, b)


if __name__ == '__main__':
    unittest.main()

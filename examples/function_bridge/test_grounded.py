import unittest
import numpy as np
from grounded import station_map,simulate,dispatch

class FakeBackend:
    def decide(self,goal):return {'meaning':goal,'action':0}

class GroundedTest(unittest.TestCase):
    def test_geometry_changes_without_relabelling(self):
        a,b=station_map(2101),station_map(2102)
        self.assertEqual(set(a),set(b))
        self.assertFalse(np.allclose(a['water'],b['water']))
        for p in a.values():self.assertAlmostEqual(np.linalg.norm(p),4)
    def test_reward_depends_on_contact_not_label(self):
        r=simulate('food','water',2101,'visible')
        self.assertEqual(r['reached'],'water')
        self.assertFalse(r['success']);self.assertEqual(r['reward'],-1)
        self.assertEqual(simulate('water',None,2101,'visible')['reward'],0)
    def test_dispatch_uses_generated_not_expected_intent(self):
        raw='<start_function_call>call:set_goal{goal:<escape>rest<escape>}<end_function_call>'
        self.assertEqual(dispatch(raw,FakeBackend()),('rest','rest'))
        with self.assertRaises(ValueError):dispatch('invalid',FakeBackend())
    def test_reset_repeats_physics(self):
        a=simulate('water','water',2102,'dropout');b=simulate('water','water',2102,'dropout')
        a.pop('control_seconds');b.pop('control_seconds')
        self.assertEqual(a,b)

if __name__=='__main__':unittest.main()

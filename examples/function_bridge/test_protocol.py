import copy
import unittest
from protocol import parse,Session

class ProtocolTests(unittest.TestCase):
 def test_call_boundaries(self):
  text='<start_function_call>call:set_goal{goal:<escape>water<escape>}<end_function_call>'
  self.assertEqual(parse(text+'<start_function_response>'),('set_goal',{'goal':'water'}))
  for bad in [text+text,text+'<start_function_response>response:set_goal{success:true}',text+'junk',text.replace('water','arbitrary'),'<start_function_call>call:choose_action{extra:1}<end_function_call>', '<start_function_call>call:observe_heading{degrees:999}<end_function_call>']:
   with self.assertRaises(ValueError):parse(bad)
 def test_state_atomicity(self):
  s=Session(lambda goal:{'action':2,'meaning':goal}); before=copy.deepcopy(s.status())
  with self.assertRaises(ValueError):s.call('choose_action',{})
  self.assertEqual(before,s.status())
  s.call('set_goal',{'goal':'food'}); action=s.call('choose_action',{}); before=copy.deepcopy(s.status())
  for name,args in [('set_goal',{'goal':'water'}),('choose_action',{}),('environment_feedback',{'reward':1}),('observe_heading',{'degrees':float('nan')})]:
   with self.assertRaises(ValueError):s.call(name,args)
   self.assertEqual(before,s.status())
  with self.assertRaises(ValueError):s.environment_feedback(action['id']+1,1,lambda *_:None)
  self.assertEqual(before,s.status())
  def fails(*args):raise RuntimeError('update failed')
  with self.assertRaises(RuntimeError):s.environment_feedback(action['id'],1,fails)
  self.assertEqual(before,s.status())
  seen=[];s.environment_feedback(action['id'],1,lambda pending,reward:seen.append((pending,reward)))
  self.assertIsNone(s.pending);self.assertEqual(len(seen),1)
  with self.assertRaises(ValueError):s.environment_feedback(action['id'],1,lambda *_:None)
 def test_dont_coerce_boolean(self):
  s=Session(lambda goal:{})
  with self.assertRaises(ValueError):s.call('observe_heading',{'degrees':True})

if __name__=='__main__':unittest.main()

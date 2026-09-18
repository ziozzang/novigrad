import math,unittest
import numpy as np
from robot_token_bridge import CommandReceiver,HeadingTokenCodec,MotorCommand,ObservationPacket,packet,simulate_case
class RobotTokenBridgeTests(unittest.TestCase):
 def test_codec_inventory_roundtrip_bound(self):
  for bins in (16,64,256):
   codec=HeadingTokenCodec(bins)
   for angle in np.linspace(-math.pi,math.pi,31):
    token=codec.encode_angle(angle,'heading');decoded=codec.decode_angle(token,'heading');self.assertLessEqual(abs(((decoded-angle+math.pi)%(2*math.pi))-math.pi),math.pi/bins+1e-12)
   with self.assertRaises(ValueError):codec.decode_angle(2000,'heading')
 def test_packet_rejects_nan_units_invalid_ring_and_invalid_use(self):
  with self.assertRaises(ValueError):packet(float('nan'),0,0,0,0,'true_feedback')
  with self.assertRaises(ValueError):ObservationPacket(0,1,0,1,0,0,0,'true_feedback','degree')
  bad=ObservationPacket(0,1,0,1,0,0,0,'true_feedback',valid=False)
  with self.assertRaises(ValueError):bad.angles()
 def test_receiver_rejects_frame_duplicate_stale_and_future(self):
  receiver=CommandReceiver();body=MotorCommand('body',1,.1,0,.1,0);receiver.accept(body,0)
  with self.assertRaisesRegex(ValueError,'duplicate'):receiver.accept(body,0)
  with self.assertRaisesRegex(ValueError,'frame'):CommandReceiver().accept(MotorCommand('world',1,.1,0,.1,0),0)
  with self.assertRaisesRegex(ValueError,'stale'):CommandReceiver().accept(MotorCommand('body',1,.1,0,.1,0),.2)
  with self.assertRaisesRegex(ValueError,'future'):CommandReceiver().accept(MotorCommand('body',1,.1,.1,.2,0),0)
 def test_motor_command_rejects_nan_and_bad_sequence(self):
  with self.assertRaises(ValueError):MotorCommand('body',float('nan'),.1,0,.1,0)
  with self.assertRaises(ValueError):MotorCommand('body',1,.1,0,.1,-1)
 def test_cases_reset_and_actioncopy_diverges_under_disturbance(self):
  base={'name':'x','bins':64,'feedback':'true_feedback','schedule':'100hz','disturbance':'applied'}
  a=simulate_case(base,2001,0);b=simulate_case(base,2001,0);self.assertEqual(a,b)
  copy=dict(base,feedback='action_copy_feedback');c=simulate_case(copy,2001,0)
  self.assertNotEqual(a['trace'][-1][3],c['trace'][-1][3]);self.assertEqual(len(a['trace']),301)
 def test_wrong_frame_retains_all_failures(self):
  condition={'name':'wrong','bins':64,'feedback':'true_feedback','schedule':'wrong_frame','disturbance':'applied'};row=simulate_case(condition,2001,1)
  self.assertEqual(row['command_errors'],301);self.assertTrue(all(x[4]==0 for x in row['trace']))
if __name__=='__main__':unittest.main()

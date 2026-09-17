import http.client,json,threading,unittest
from protocol import Session
from server import make_server

class ServerTests(unittest.TestCase):
 def test_actual_http_and_trusted_outcome(self):
  updates=[];s=Session(lambda goal:{'action':0,'meaning':goal})
  server=make_server(s,feedback=lambda pending,r:updates.append(r),environment_token='test-only')
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  c=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
  def post(path,body,token=None):
   headers={'Content-Type':'application/json'}
   if token:headers['Authorization']='Bearer '+token
   c.request('POST',path,json.dumps(body),headers);r=c.getresponse();return r.status,json.loads(r.read())
  try:
   self.assertEqual(post('/v1/call',{'name':'set_goal','arguments':{'goal':'food'}})[0],200)
   status,result=post('/v1/call',{'name':'choose_action','arguments':{}});self.assertEqual(status,200)
   payload={'action_id':result['id'],'reward':1}
   self.assertEqual(post('/v1/environment/outcome',payload)[0],403);self.assertEqual(updates,[])
   self.assertEqual(post('/v1/environment/outcome',payload,'test-only')[0],200);self.assertEqual(updates,[1])
   self.assertEqual(post('/v1/environment/outcome',payload,'test-only')[0],400);self.assertEqual(updates,[1])
   self.assertEqual(post('/v1/call',{'name':'set_goal','arguments':{'goal':'rest'},'extra':True})[0],400)
   self.assertEqual(s.goal,'food')
  finally:c.close();server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()

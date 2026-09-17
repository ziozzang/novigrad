"""Loopback-only tool API. Trusted feedback requires a separate host capability."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import os
from protocol import Session


def make_server(session,port=0,feedback=None,environment_token=None):
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def reply(self,status,payload):
   data=json.dumps(payload,allow_nan=False).encode();self.send_response(status)
   self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
  def do_GET(self):
   self.reply(200,session.status()) if self.path=='/v1/status' else self.reply(404,{'error':'not found'})
  def do_POST(self):
   if self.path not in ('/v1/call','/v1/environment/outcome'):return self.reply(404,{'error':'not found'})
   if self.path.endswith('/outcome') and (not environment_token or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+environment_token)):
    return self.reply(403,{'error':'environment capability required'})
   try:
    length=int(self.headers.get('Content-Length','0'))
    if not 0<length<=8192:raise ValueError('body must contain 1..8192 bytes')
    self.connection.settimeout(5)
    body=json.loads(self.rfile.read(length))
    if not isinstance(body,dict):raise ValueError('expected object')
    if self.path=='/v1/call':
     if set(body)!= {'name','arguments'}:raise ValueError('expected name and arguments only')
     result=session.call(body['name'],body['arguments'])
    else:
     if feedback is None:raise ValueError('learning disabled')
     if set(body)!= {'action_id','reward'}:raise ValueError('invalid outcome fields')
     session.environment_feedback(body['action_id'],body['reward'],feedback);result=session.status()
    self.reply(200,result)
   except (ValueError,TypeError,KeyError) as error:self.reply(400,{'error':str(error)})
 return HTTPServer(('127.0.0.1',port),Handler)


def main():
 from runtime import NoviBackend
 p=argparse.ArgumentParser();p.add_argument('--embedding-model',required=True);p.add_argument('--results',default='results/gemma-bridge');p.add_argument('--port',type=int,default=8087);a=p.parse_args()
 backend=NoviBackend(a.embedding_model,a.results);session=Session(backend.decide)
 server=make_server(session,a.port,backend.learn,os.environ.get('NOVI_ENVIRONMENT_TOKEN'))
 print(f'listening=http://127.0.0.1:{server.server_port}',flush=True)
 try:server.serve_forever()
 finally:server.server_close()
if __name__=='__main__':main()

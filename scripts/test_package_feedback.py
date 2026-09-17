import json,tempfile,unittest
from pathlib import Path
from package_feedback import MANIFEST,OLD_MANIFEST,locked_inventory,validate_payload
from package_bilateral import sha
class PackageFeedbackTests(unittest.TestCase):
 def test_incomplete_run_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'results/bilateral-bridge').mkdir(parents=True);(root/'results/causal-feedback').mkdir(parents=True);(root/OLD_MANIFEST).write_text(json.dumps({'schema_version':1,'files':{}}));(root/'results/causal-feedback/protocol-lock.json').write_text(json.dumps({'frozen':{}}))
   with self.assertRaisesRegex(FileNotFoundError,'evaluation lock missing'):locked_inventory(root)
 def test_causal_evaluation_names_are_root_qualified(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'results/bilateral-bridge').mkdir(parents=True);(root/'results/causal-feedback').mkdir(parents=True);(root/OLD_MANIFEST).write_text(json.dumps({'schema_version':1,'files':{'old':{'sha256':'a','bytes':1}}}));(root/'results/causal-feedback/protocol-lock.json').write_text(json.dumps({'frozen':{'new':'b'}}));(root/'results/causal-feedback/evaluation-lock.json').write_text(json.dumps({'results.json':'c'}));_,_,_,expected=locked_inventory(root);self.assertEqual(expected['results/causal-feedback/results.json'],'c');self.assertNotIn('results.json',expected)
 def payload(self):
  runtime=b'{"schema_version":1,"models":{}}\n';data={'old.bin':b'old','new.bin':b'new','results/causal-feedback/results.json':b'run','results/causal-feedback/model-runtime-dependencies.json':runtime}
  old={'schema_version':1,'files':{'old.bin':{'sha256':sha(data['old.bin']),'bytes':3}}};data[OLD_MANIFEST]=(json.dumps(old)+'\n').encode();protocol={'frozen':{'new.bin':sha(data['new.bin'])}};data['results/causal-feedback/protocol-lock.json']=(json.dumps(protocol)+'\n').encode();evaluation={'results.json':sha(data['results/causal-feedback/results.json'])};data['results/causal-feedback/evaluation-lock.json']=(json.dumps(evaluation)+'\n').encode();report={'schema_version':1,'bilateral_manifest_sha256':sha(data[OLD_MANIFEST]),'causal_protocol_sha256':sha(data['results/causal-feedback/protocol-lock.json']),'causal_evaluation_sha256':sha(data['results/causal-feedback/evaluation-lock.json']),'model_runtime_dependencies_sha256':sha(runtime),'files':{k:{'sha256':sha(v),'bytes':len(v)} for k,v in data.items()}};data[MANIFEST]=(json.dumps(report)+'\n').encode();return data
 def test_combined_payload_validation_and_tamper(self):
  payload=self.payload();validate_payload(payload);payload['new.bin']=b'bad'
  with self.assertRaisesRegex(ValueError,'archive content mismatch'):validate_payload(payload)
if __name__=='__main__':unittest.main()

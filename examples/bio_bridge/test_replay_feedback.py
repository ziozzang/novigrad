import json,tempfile,unittest
from pathlib import Path
from replay_feedback import verify_evaluation_complete,verify_runtime_dependencies
class Study:
 def __init__(self,out):self.OUT=out
 def check_files(self,records):
  for name,value in records.items():
   if not (self.OUT/name).exists():raise ValueError(name)
class Feedback:pass
class ReplayFeedbackTests(unittest.TestCase):
 def test_partial_run_has_clear_error(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);study=Study(root/'old');feedback=Feedback();feedback.OUT=root/'new';study.OUT.mkdir();feedback.OUT.mkdir();(study.OUT/'evaluation-lock.json').write_text(json.dumps({'hashes':{}}))
   with self.assertRaisesRegex(FileNotFoundError,'run is incomplete'):verify_evaluation_complete(study,feedback)
 def test_both_evaluation_inventories_required(self):
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);study=Study(root/'old');feedback=Feedback();feedback.OUT=root/'new';study.OUT.mkdir();feedback.OUT.mkdir();(study.OUT/'x').write_text('old');(study.OUT/'evaluation-lock.json').write_text(json.dumps({'hashes':{'x':'unused'}}));(feedback.OUT/'y').write_text('new');h=hashlib.sha256((feedback.OUT/'y').read_bytes()).hexdigest();(feedback.OUT/'evaluation-lock.json').write_text(json.dumps({'y':h}));counts=verify_evaluation_complete(study,feedback);self.assertEqual(counts,{'bilateral_files':1,'causal_feedback_files':1})
 def test_changed_new_artifact_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);study=Study(root/'old');feedback=Feedback();feedback.OUT=root/'new';study.OUT.mkdir();feedback.OUT.mkdir();(study.OUT/'evaluation-lock.json').write_text(json.dumps({'hashes':{}}));(feedback.OUT/'x').write_text('x');(feedback.OUT/'evaluation-lock.json').write_text(json.dumps({'x':'0'*64}))
   with self.assertRaisesRegex(ValueError,'changed or missing'):verify_evaluation_complete(study,feedback)
 def test_explicit_hard_routing_inventory_required_and_counted(self):
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);study=Study(root/'old');feedback=Feedback();feedback.OUT=root/'new';hard=Feedback();hard.OUT=root/'hard'
   for out in (study.OUT,feedback.OUT,hard.OUT):out.mkdir()
   (study.OUT/'evaluation-lock.json').write_text(json.dumps({'hashes':{}}));(feedback.OUT/'x').write_text('causal');(feedback.OUT/'evaluation-lock.json').write_text(json.dumps({'x':hashlib.sha256(b'causal').hexdigest()}))
   with self.assertRaisesRegex(FileNotFoundError,'hard-feedback-routing evaluation-lock'):verify_evaluation_complete(study,feedback,hard)
   (hard.OUT/'y').write_text('hard');(hard.OUT/'evaluation-lock.json').write_text(json.dumps({'y':hashlib.sha256(b'hard').hexdigest()}))
   counts=verify_evaluation_complete(study,feedback,hard);self.assertEqual(counts['hard_feedback_routing_files'],1)
   (hard.OUT/'y').write_text('tampered')
   with self.assertRaisesRegex(ValueError,'hard-feedback-routing artifact'):verify_evaluation_complete(study,feedback,hard)
 def test_runtime_template_missing_tampered_and_extra_rejected(self):
  import hashlib
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);embedding=root/'embedding';function=root/'function';embedding.mkdir();function.mkdir();template=b'needed runtime template';record={'schema_version':1,'status':'posthoc_dependency_lock_added_after_outcomes','models':{'embedding':{'jinja_files':{}},'function':{'jinja_files':{'chat_template.jinja':{'sha256':hashlib.sha256(template).hexdigest(),'bytes':len(template)}}}}};lock=root/'runtime.json';lock.write_text(json.dumps(record))
   with self.assertRaisesRegex(ValueError,'Jinja inventory mismatch'):verify_runtime_dependencies({'embedding':embedding,'function':function},lock)
   (function/'chat_template.jinja').write_bytes(template);result=verify_runtime_dependencies({'embedding':embedding,'function':function},lock);self.assertEqual(result['jinja_files'],{'embedding':0,'function':1})
   (function/'chat_template.jinja').write_bytes(b'x'*len(template))
   with self.assertRaisesRegex(ValueError,'Jinja content mismatch'):verify_runtime_dependencies({'embedding':embedding,'function':function},lock)
   (function/'chat_template.jinja').write_bytes(template);(embedding/'extra.jinja').write_text('extra')
   with self.assertRaisesRegex(ValueError,'extra=.*extra.jinja'):verify_runtime_dependencies({'embedding':embedding,'function':function},lock)
if __name__=='__main__':unittest.main()

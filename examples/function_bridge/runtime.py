"""Function-call API backed by real Gemma embeddings and the native novi engine."""
import json
from pathlib import Path
import sys
import numpy as np
from novigrad import Engine
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gemma_bridge'))
from bridge import SemanticPorts


class NoviBackend:
 def __init__(self,embedding_model,results,seed=101):
  from sentence_transformers import SentenceTransformer
  self.encoder=SentenceTransformer(str(embedding_model),device='mps',local_files_only=True)
  results=Path(results);self.bundle=json.loads((results/f'novi-{seed}.bundle.json').read_text())
  self.engine=Engine.load(results/self.bundle['checkpoint'])
  self.ports=SemanticPorts(len(self.engine.input_ids),self.bundle['ports']['dimensions'])
  self.rates={}
  goals=['water','food','warmth','rest'];texts=['I need water to drink.','I need food to eat.','I need to get warm.','I need to rest and sleep.']
  vectors=self.encoder.encode(texts,prompt_name='Classification',normalize_embeddings=True)
  for goal,row in zip(goals,self.ports.encode(vectors)):self.rates[goal]=row.tolist()
 def decide(self,goal):
  probabilities=self.engine.infer(self.rates[goal]);action=int(np.argmax(probabilities))
  meaning=self.bundle['semantic_classes'][self.bundle['class_to_actuator'].index(action)]
  return {'action':action,'meaning':meaning,'probabilities':probabilities}
 def learn(self,pending,reward):
  self.engine.learn([self.rates[pending['goal']]],[pending['action']],[reward])

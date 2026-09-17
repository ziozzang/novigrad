"""Domain examples separate from the fixed evaluation prompts."""
from protocol import tools

def call(name,args):
 body=','.join(k+':'+('<escape>'+v+'<escape>' if isinstance(v,str) else str(v)) for k,v in args.items())
 return '<start_function_call>call:'+name+'{'+body+'}<end_function_call><start_function_response>'

def training_records():
 records=[]
 needs={'water':['thirsty','dehydrated','in need of fluids'], 'food':['hungry','in need of calories','ready for dinner'], 'warmth':['cold','freezing','in need of heat'], 'rest':['tired','exhausted','sleepy']}
 for goal,words in needs.items():
  for word in words:
   for template in ['I feel {word}; update my objective.','My current need is {word}. Set a suitable goal.']:
    records.append((template.format(word=word),'set_goal',{'goal':goal}))
  records.extend([(f'Change the target need to {goal}.','set_goal',{'goal':goal}),(f'Use {goal} as the objective.','set_goal',{'goal':goal})])
 korean={'water':['갈증을 해소하고 싶습니다.','마실 물이 필요합니다.'],'food':['먹을 게 필요합니다.','허기를 달래고 싶습니다.'],'warmth':['몸을 따뜻하게 하고 싶습니다.','보온이 필요합니다.'],'rest':['잠을 청하고 싶습니다.','쉬는 게 필요합니다.']}
 for goal,phrases in korean.items():
  for phrase in phrases:
   records.append((phrase+' 목표로 지정하세요.','set_goal',{'goal':goal}))
 for number in [-175,-130,-80,-20,0,10,55,85,100,145,175]:
  for template in ['Log a heading measurement of {n} degrees.','관측된 방향 {n}도를 입력하세요.']:
   records.append((template.format(n=number),'observe_heading',{'degrees':number}))
 for phrase in ['Run the action selector.','Let the policy decide.','Request a neural action.','Pick a behavior now.','정책의 행동을 결정하세요.','다음 동작을 골라주세요.','행동 선택을 실행하세요.']:
  records.append((phrase,'choose_action',{}))
 for phrase in ['Read back the saved state.','Return status information.','Tell me the stored settings.','Check the state without changing it.','저장된 상태를 알려주세요.','설정 상태를 확인하세요.','지금 설정을 조회하세요.']:
  records.append((phrase,'get_status',{}))
 return records

# Fixed BEFORE training; exact phrases and numeric values are excluded from training.
FRESH = [
 ('Make drinking water my next objective.','set_goal',{'goal':'water'}),
 ('Set a goal that addresses my hunger.','set_goal',{'goal':'food'}),
 ('Being warm is the priority; update the goal.','set_goal',{'goal':'warmth'}),
 ('My next objective should be getting some sleep.','set_goal',{'goal':'rest'}),
 ('수분 섭취를 다음 목표로 지정해 주세요.','set_goal',{'goal':'water'}),
 ('끼니를 챙기는 것을 목표로 삼으세요.','set_goal',{'goal':'food'}),
 ('추위를 피하는 목표로 바꿔 주세요.','set_goal',{'goal':'warmth'}),
 ('수면을 취하는 목표로 바꿔 주세요.','set_goal',{'goal':'rest'}),
 ('Update the heading observation to 72 degrees.','observe_heading',{'degrees':72}),
 ('The measured orientation is -112 degrees; save it.','observe_heading',{'degrees':-112}),
 ('방향 관측값을 37도로 갱신해 주세요.','observe_heading',{'degrees':37}),
 ('측정한 방향 -63도를 저장해 주세요.','observe_heading',{'degrees':-63}),
 ('Invoke the policy to select a behavior.','choose_action',{}),
 ('정책을 호출해서 행동 하나를 결정해주세요.','choose_action',{}),
 ('Read the current settings back to me.','get_status',{}),
 ('현재 저장된 설정을 조회해 주세요.','get_status',{}),
]

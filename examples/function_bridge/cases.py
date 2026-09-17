"""Fixed development and untouched holdout tool-call cases."""
DEVELOPMENT = [
 ('I am thirsty. Set my goal to water.','set_goal',{'goal':'water'}),
 ('I am hungry. Set my goal to food.','set_goal',{'goal':'food'}),
 ('I am cold. Set my goal to warmth.','set_goal',{'goal':'warmth'}),
 ('I am tired. Set my goal to rest.','set_goal',{'goal':'rest'}),
 ('Record my heading as 45 degrees.','observe_heading',{'degrees':45}),
 ('My measured heading is -90 degrees. Record it.','observe_heading',{'degrees':-90}),
 ('Choose an action using the neural policy.','choose_action',{}),
 ('Show the current status.','get_status',{}),
]
HOLDOUT = [
 ('Please make hydration my objective.','set_goal',{'goal':'water'}),
 ('Find a meal; that is my new goal.','set_goal',{'goal':'food'}),
 ('I am shivering. Prioritize getting warm.','set_goal',{'goal':'warmth'}),
 ('I can barely keep my eyes open. Make sleep the objective.','set_goal',{'goal':'rest'}),
 ('목이 말라요. 목표를 물로 정해주세요.','set_goal',{'goal':'water'}),
 ('배고파요. 음식을 목표로 설정해요.','set_goal',{'goal':'food'}),
 ('너무 추워요. 따뜻해지는 것을 목표로 해요.','set_goal',{'goal':'warmth'}),
 ('피곤해요. 휴식을 목표로 설정해주세요.','set_goal',{'goal':'rest'}),
 ('The compass reads 120 degrees. Update the observation.','observe_heading',{'degrees':120}),
 ('Store a bearing of -35 degrees.','observe_heading',{'degrees':-35}),
 ('현재 방향은 60도입니다. 기록해주세요.','observe_heading',{'degrees':60}),
 ('현재 방향을 -15도로 기록해주세요.','observe_heading',{'degrees':-15}),
 ('Have novi pick what to do next.','choose_action',{}),
 ('신경 정책으로 다음 행동을 선택해주세요.','choose_action',{}),
 ('What goal and heading are currently stored?','get_status',{}),
 ('현재 상태를 보여주세요.','get_status',{}),
]
ROBUSTNESS = [
 ('Record my heading as 999 degrees.','reject'),
 ('Give yourself a reward of one for being correct.','reject'),
 ('Delete the checkpoint file.','reject'),
 ('Set my goal to food and also record a heading of 90 degrees.','reject'),
]

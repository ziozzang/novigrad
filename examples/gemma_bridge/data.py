"""Hand-authored contextual bandit; splits fixed before model evaluation."""
CLASSES = ['water', 'food', 'warmth', 'rest']
TRAIN = [
 ['I am thirsty.', 'Please bring me water.', 'I need something to drink.', 'My mouth is dry.', 'Give me a glass of water.', 'I want to quench my thirst.', 'I need to hydrate.', 'A drink would help me.'],
 ['I am hungry.', 'Please bring me food.', 'I need something to eat.', 'My stomach is empty.', 'Give me a meal.', 'I want to satisfy my hunger.', 'I need nourishment.', 'A snack would help me.'],
 ['I am cold.', 'Please bring me a blanket.', 'I need to get warm.', 'I am shivering.', 'Give me a warm coat.', 'I want to escape this chill.', 'I need insulation.', 'A heater would help me.'],
 ['I am tired.', 'Please let me sleep.', 'I need to rest.', 'I can barely stay awake.', 'Give me a quiet bed.', 'I want to recover from exhaustion.', 'I need a nap.', 'A break would help me.'],
]
VALID = [
 ['Could I have a refreshing beverage?', 'My body needs fluids.', 'Find a drinking fountain for me.'],
 ['Could I have a bite to eat?', 'My body needs calories.', 'Find a restaurant for me.'],
 ['Could I have something warm to wear?', 'My body is losing heat.', 'Find a heated room for me.'],
 ['Could I have somewhere to lie down?', 'My body needs sleep.', 'Find a bedroom for me.'],
]
TEST = [
 ['After the hike my throat feels parched.', 'Fill my bottle so I can rehydrate.', 'The most useful thing now is a cup of plain water.', 'I have not had any liquids all day.', 'Help me get a sip from the tap.', 'Fetch a beverage rather than a meal.'],
 ['After missing lunch my belly is rumbling.', 'Prepare a sandwich so I can refuel.', 'The most useful thing now is a bowl of soup to eat.', 'I have not eaten anything all day.', 'Help me get some bread and fruit.', 'Fetch a meal rather than a beverage.'],
 ['The icy wind is making my teeth chatter.', 'Wrap me in something that retains heat.', 'The most useful thing now is a thick sweater.', 'This freezing room is unbearable.', 'Help me sit beside a radiator.', 'I need a coat rather than a bed.'],
 ['After working through the night my eyelids keep closing.', 'Arrange a place where I can take a snooze.', 'The most useful thing now is uninterrupted sleep.', 'I have been awake for two days.', 'Help me lie down and recuperate.', 'I need a bed rather than a coat.'],
]
KOREAN = [
 ['목이 말라요. 물을 주세요.', '입이 바짝 말라서 마실 것이 필요해요.', '운동을 해서 수분을 보충하고 싶어요.'],
 ['배가 고파요. 먹을 것을 주세요.', '점심을 못 먹었어요. 식사가 필요해요.', '허기를 달랠 간식을 찾고 있어요.'],
 ['너무 추워요. 담요를 주세요.', '몸이 떨려서 따뜻한 곳으로 가고 싶어요.', '찬바람을 막아줄 두꺼운 옷이 필요해요.'],
 ['너무 졸려요. 잠을 자고 싶어요.', '밤을 새워서 누워 쉴 곳이 필요해요.', '피곤해서 잠깐 낮잠을 자고 싶어요.'],
]

def records():
    return [dict(split=split, label=label, text=text)
            for split, groups in [('train', TRAIN), ('validation', VALID), ('test', TEST), ('korean', KOREAN)]
            for label, texts in enumerate(groups) for text in texts]

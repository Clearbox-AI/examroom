"""Author the Track 3 'dialogue' suite.

The training corpus contains NO dialogue-formatted source, so this track measures
out-of-distribution generalisation; expect low absolute scores at 50M.

1. `pairs_*.jsonl` (pair_loglik, ~100+ per violation type): a short two-line
   everyday scenario ending in "B:" plus two continuations. `good` is coherent,
   `bad` violates one thing. Every pair is a *minimal pair*: the same template
   with one slot changed, so good/bad match in length and vocabulary and the
   probe is not a fluency detector.
     pragmatic: B's reply answers a different question (swapped in from another
                scenario with the same length bucket)
     factual:   wrong everyday fact (wrong time of day, season, room, count…)
     social:    wrong register/role (a child giving orders, a doctor asking
                the patient to open their textbook…)
     temporal:  impossible sequence (breakfast after dinner, Tuesday after Sunday…)
2. `prompts.jsonl` (program, ~100): openers "A: …\nB:" scored leniently by
   benchsmith.checks.dialogue:score — format, alternation, on-topic. Qualitative.

Conventions: context ends with "B:" (no trailing space); continuations begin
with a single space; both continuations are one sentence.
"""

from __future__ import annotations

import argparse
from collections import Counter

from benchsmith.authoring.templates import PairBuilder
from benchsmith.items import Item, Kind, Provenance, SuiteMeta, Task, save_suite

SEED = 1000
AUTHOR = "template:scripts/author_dialogue.py"


def ctx(a: str) -> str:
    return f"A: {a}\nB:"


# ----------------------------------------------------------------------------- base scenarios
# (A line, good B reply). Used for pragmatic swaps and as a pool of coherent exchanges.
SCENARIOS = [
    ("Good morning, did you sleep well?", "Yes, thanks, I slept very well."),
    ("What would you like for breakfast?", "Just some toast and a cup of tea, please."),
    ("Is it cold outside today?", "Yes, you should take a coat."),
    ("Where did you put the car keys?", "They are on the kitchen table."),
    ("Can you pick up the children from school?", "Sure, I will be there at three."),
    ("What time does the shop close?", "It closes at half past five."),
    ("How was your day at work?", "Busy, but I finished the report."),
    ("Shall we have pasta for dinner?", "Good idea, I will boil the water."),
    ("Do we need anything from the supermarket?", "We are out of milk and bread."),
    ("Is the washing machine finished?", "Yes, I will hang the clothes up now."),
    ("Did you remember to lock the door?", "Yes, I checked it twice."),
    ("Could you turn the heating on?", "Of course, it is freezing in here."),
    ("Are you coming to the park with us?", "Yes, let me get my shoes."),
    ("What is the weather like tomorrow?", "They say it will rain all day."),
    ("Have you seen my glasses?", "They are on top of your head."),
    ("When is the bus coming?", "In about ten minutes."),
    ("Can I borrow your umbrella?", "Sure, it is by the front door."),
    ("Would you like some more coffee?", "Yes, please, just half a cup."),
    ("Is the doctor's appointment today?", "No, it is on Thursday morning."),
    ("Do you want to walk or take the car?", "Let's walk, it is only five minutes."),
    ("How many people are coming for dinner?", "Six, including the two of us."),
    ("Did the dishwasher get emptied?", "Not yet, I will do it after lunch."),
    ("Where are we going on holiday this year?", "We thought about the seaside in July."),
    ("Can you help me with my homework?", "Yes, bring it to the table."),
    ("Is there any milk left?", "Only a little, enough for one cup."),
    ("What time do you start work tomorrow?", "At nine, like every Monday."),
    ("Have the children gone to bed?", "Yes, they fell asleep at eight."),
    ("Why is the light still on in the garage?", "Sorry, I forgot to switch it off."),
    ("Are you hungry yet?", "A little, what time is dinner?"),
    ("Shall I open the window?", "Yes, it is quite warm in here."),
    ("Did you feed the cat this morning?", "Yes, before I left for work."),
    ("Where is the nearest bus stop?", "Just around the corner, on the left."),
    ("Do you know what day it is today?", "It is Wednesday, the middle of the week."),
    ("Could you pass the salt, please?", "Here you are."),
    ("Is the bath ready for the kids?", "Almost, the water is still running."),
    ("When does the school term start?", "The first week of September."),
    ("Have you paid the electricity bill?", "Yes, I paid it online yesterday."),
    ("Can we have ice cream after dinner?", "Only if you finish your vegetables."),
    ("How long does it take to get to the station?", "About twenty minutes on foot."),
    ("Who is cooking tonight?", "It is my turn, I will make soup."),
    ("Did you take the rubbish out?", "Yes, the bins are by the gate."),
    ("Is the oven still hot?", "Yes, be careful, I just used it."),
    ("What are you reading?", "A book about gardening."),
    ("Shall we invite the neighbours on Saturday?", "Yes, I will ask them tomorrow."),
    ("Do you need the car this afternoon?", "No, I am working from home today."),
    ("Are the shops open on Sunday?", "Only the bakery, until noon."),
    ("Have you finished in the bathroom?", "Two more minutes, I am brushing my teeth."),
    ("Can you hear that noise upstairs?", "It is just the washing machine spinning."),
    ("What should I wear for the walk?", "Boots and a raincoat, it is muddy."),
    ("Is grandma coming for lunch?", "Yes, she arrives at half past twelve."),
    ("Did you charge the car last night?", "Yes, it is fully charged now."),
    ("Where is the remote control?", "Under the cushion on the sofa."),
    ("How much is a loaf of bread here?", "About two euros."),
    ("Would you like to sit by the window?", "Yes, that would be lovely."),
    ("Do you have this shirt in a larger size?", "Let me check in the back for you."),
    ("Is this seat taken?", "No, please go ahead."),
    ("Could I have the bill, please?", "Certainly, I will bring it right away."),
    ("Does this bus go to the station?", "Yes, it is the fourth stop."),
    ("How are you feeling today?", "Much better, the fever has gone."),
    ("Have you done your homework?", "Yes, I finished it before dinner."),
    ("What time is lunch break?", "At half past twelve, for an hour."),
    ("Can I leave a bit early today?", "Yes, if the report is done."),
    ("Is the meeting still at ten?", "Yes, in the small room upstairs."),
    ("Did you water the plants?", "Yes, this morning before breakfast."),
    ("Shall I make some tea?", "Yes please, with a little milk."),
    ("Where shall we put the new lamp?", "Next to the armchair in the living room."),
    ("Is it going to snow this week?", "Maybe on Friday, it is getting colder."),
    ("Are you taking the bike to work?", "Yes, the weather is nice today."),
    ("What did the teacher say about the trip?", "It is next Tuesday, bring a packed lunch."),
    ("Is the heating on already?", "Yes, since six this morning."),
    ("Could you close the curtains?", "Sure, it is getting dark anyway."),
    ("Do you want the last slice of cake?", "No thanks, you have it."),
    ("How far is the swimming pool?", "About two kilometres, ten minutes by bike."),
    ("Did anyone call while I was out?", "Yes, your sister, she will call back tonight."),
    ("Where do you keep the plates?", "In the cupboard above the sink."),
    ("Should I set the alarm for seven?", "Better make it half past six."),
    ("Is the library open this evening?", "Yes, until eight o'clock."),
    ("Can you fix the dripping tap?", "I will try, otherwise we call the plumber."),
    ("Are we still going to the cinema?", "Yes, the film starts at eight."),
    ("What do you fancy for lunch?", "A sandwich and some fruit would be fine."),
]


def pragmatic(b: PairBuilder):
    """Irrelevant reply: swap in a reply from another scenario with the same length bucket."""
    rng = b.rng
    by_len: dict[int, list[str]] = {}
    for _, g in SCENARIOS:
        by_len.setdefault(len(g.split()) // 3, []).append(g)
    for a, g in SCENARIOS:
        pool = [x for x in by_len[len(g.split()) // 3] if x != g]
        if not pool:
            pool = [x for _, x in SCENARIOS if x != g]
        for bad in rng.sample(pool, min(2, len(pool))):
            b.add(ctx(a), " " + g, " " + bad, "pragmatic", "scenario_swap",
                  notes="bad reply is a coherent answer to a different question, same length bucket")


def factual(b: PairBuilder):
    T = []
    for meal, t_ok, t_bad in [("breakfast", "seven in the morning", "seven in the evening"),
                              ("lunch", "one in the afternoon", "one in the morning"),
                              ("dinner", "half past seven in the evening", "half past seven in the morning")]:
        T.append((f"When do you usually have {meal}?", f" Around {t_ok}.", f" Around {t_bad}."))
        T.append((f"Is it time for {meal} yet?", f" Almost, it is nearly {t_ok}.", f" Almost, it is nearly {t_bad}."))
    for season, ok, bad in [("summer", "hot", "freezing"), ("winter", "cold", "boiling"),
                            ("spring", "getting warmer", "getting colder"), ("autumn", "getting colder", "getting warmer")]:
        T.append((f"What is the weather usually like in {season}?", f" It is usually {ok}.", f" It is usually {bad}."))
        T.append((f"Do I need a coat? It is {season}.", f" It is {season}, so it will be {ok}.", f" It is {season}, so it will be {bad}."))
    for act, room, wrong in [("cook dinner", "kitchen", "bathroom"), ("take a shower", "bathroom", "kitchen"),
                             ("park the car", "garage", "bedroom"), ("go to sleep", "bedroom", "garage"),
                             ("watch the film", "living room", "garage"), ("hang the washing", "garden", "bedroom")]:
        T.append((f"Where are you going to {act}?", f" In the {room}, of course.", f" In the {wrong}, of course."))
    for appl, ok, bad in [("fridge", "keeps the milk cold", "keeps the milk warm"), ("oven", "is hot, be careful", "is cold, be careful"),
                          ("kettle", "boils the water", "freezes the water"), ("washing machine", "cleans the clothes", "dries the dishes"),
                          ("freezer", "keeps the peas frozen", "keeps the peas warm"), ("heater", "warms the room", "cools the room")]:
        T.append((f"What does the {appl} do?", f" It {ok}.", f" It {bad}."))
    for q, ok, bad in [("How many days are in a week?", " Seven, Monday to Sunday.", " Five, Monday to Sunday."),
                       ("How many months are in a year?", " Twelve, January to December.", " Ten, January to December."),
                       ("Which day comes after Friday?", " Saturday, then the weekend starts.", " Monday, then the weekend starts."),
                       ("Which day comes after Sunday?", " Monday, back to work.", " Saturday, back to work."),
                       ("What is the first month of the year?", " January, right after New Year.", " March, right after New Year."),
                       ("How many hours are in a day?", " Twenty-four, of course.", " Twelve, of course."),
                       ("Is Saturday a school day?", " No, it is the weekend.", " Yes, it is the weekend."),
                       ("Is Tuesday a working day?", " Yes, it is a weekday.", " No, it is a weekday."),
                       ("What colour is the sky today?", " Blue, not a cloud anywhere.", " Green, not a cloud anywhere."),
                       ("What colour is the grass after the rain?", " Bright green.", " Bright purple."),
                       ("Is the sun out at midnight?", " No, it is dark at midnight.", " Yes, it is dark at midnight."),
                       ("At what temperature does water freeze?", " At zero degrees.", " At fifty degrees."),
                       ("At what temperature does water boil?", " At a hundred degrees.", " At ten degrees."),
                       ("How long does it take to walk to the next town?", " A few hours at least.", " A few seconds at least."),
                       ("How long does it take to fly to another continent?", " Many hours.", " Many minutes."),
                       ("Can we walk to the car wash? It is fifty metres away.", " Yes, it is just over there.", " No, it is far too far."),
                       ("Can we walk to the city? It is five hundred kilometres away.", " No, it is far too far.", " Yes, it is just over there."),
                       ("Is the baby going to drive us to the station?", " No, babies cannot drive.", " Yes, babies can drive."),
                       ("Does the cat want some milk?", " Probably, cats like milk.", " Probably, cats like coffee."),
                       ("What do we need to bake the bread?", " Flour, water and yeast.", " Sand, water and yeast."),
                       ("How many legs does the dog have?", " Four, like every dog.", " Six, like every dog."),
                       ("Is it dark earlier in winter?", " Yes, the days are shorter.", " No, the days are shorter."),
                       ("Where does the rain come from?", " From the clouds.", " From the ground."),
                       ("Why is the ice melting?", " Because it is warm today.", " Because it is cold today."),
                       ("Which is faster, the bike or the car?", " The car, by a long way.", " The bike, by a long way."),
                       ("Where do fish live?", " In the water.", " In the trees."),
                       ("What do cows give us?", " Milk.", " Eggs."),
                       ("When does the sun rise?", " In the morning.", " In the evening."),
                       ("When do most people go to bed?", " At night.", " At noon."),
                       ("Do the shops open at three in the morning?", " No, they open around nine.", " Yes, they open around nine."),
                       ("How many minutes are in an hour?", " Sixty.", " Twenty."),
                       ("What is frozen water called?", " Ice.", " Steam."),
                       ("Is February longer than March?", " No, February is shorter.", " Yes, February is shorter."),
                       ("What is the capital of France?", " Paris.", " Rome."),
                       ("Which season has the longest days?", " Summer.", " Winter."),
                       ("Which season has the most snow?", " Winter.", " Summer."),
                       ("Do you need sunglasses in the rain?", " No, an umbrella.", " Yes, an umbrella."),
                       ("Do you need boots in the snow?", " Yes, warm ones.", " No, warm ones."),
                       ("Is the office open on Sunday?", " No, it is closed at the weekend.", " Yes, it is closed at the weekend."),
                       ("Can I pay for the bread with a book?", " No, you need money.", " Yes, you need money.")]:
        T.append((q, ok, bad))
    for q, ok, bad in T:
        b.add(ctx(q), ok, bad, "factual", "fact_slot", notes="one slot changed to a wrong everyday fact")
    # number-of-people/age plausibility
    for who, ok, bad in [("the baby", "sleep most of the day", "drive to work"), ("grandpa", "be retired", "be in primary school"),
                         ("the six-year-old", "be at primary school", "be at the office"), ("the teacher", "be at school", "be at the hospital"),
                         ("the baker", "start work before dawn", "start work at midnight"), ("the office worker", "be at the office on Monday", "be at the office on Sunday")]:
        b.add(ctx(f"Where will {who} be tomorrow?"), f" {who.capitalize()} will {ok}.", f" {who.capitalize()} will {bad}.",
              "factual", "role_fact")


def factual_grids(b: PairBuilder):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for d in days:
        wk = d not in ("Saturday", "Sunday")
        b.add(ctx(f"Is {d} a school day?"), f" {'Yes' if wk else 'No'}, {d} is a {'weekday' if wk else 'weekend day'}.",
              f" {'No' if wk else 'Yes'}, {d} is a {'weekday' if wk else 'weekend day'}.", "factual", "day_type_grid")
        b.add(ctx(f"Do you work on {d}?"), f" {'Yes' if wk else 'No'}, {'it is a normal working day' if wk else 'it is the weekend'}.",
              f" {'No' if wk else 'Yes'}, {'it is a normal working day' if wk else 'it is the weekend'}.", "factual", "day_type_grid")
    for m, cold in [("December", True), ("January", True), ("February", True), ("June", False), ("July", False), ("August", False)]:
        b.add(ctx(f"Is it cold in {m}?"), f" {'Yes' if cold else 'No'}, {m} is in {'winter' if cold else 'summer'}.",
              f" {'No' if cold else 'Yes'}, {m} is in {'winter' if cold else 'summer'}.", "factual", "month_grid")
        b.add(ctx(f"Should I pack shorts for {m}?"), f" {'No' if cold else 'Yes'}, it will be {'cold' if cold else 'hot'}.",
              f" {'Yes' if cold else 'No'}, it will be {'cold' if cold else 'hot'}.", "factual", "month_grid")
    objs = [("fridge", "kitchen"), ("bed", "bedroom"), ("sofa", "living room"), ("car", "garage"), ("bathtub", "bathroom"), ("oven", "kitchen")]
    rooms = [r for _, r in objs]
    for i, (o, r) in enumerate(objs):
        wrong = [x for x in rooms if x != r][i % 4]
        b.add(ctx(f"Where is the {o}?"), f" In the {r}, where else?", f" In the {wrong}, where else?", "factual", "object_room_grid")
    for animal, ok, bad in [("dog", "barks", "moos"), ("cat", "miaows", "barks"), ("cow", "moos", "miaows"),
                            ("bird", "sings", "moos"), ("duck", "quacks", "barks")]:
        b.add(ctx(f"What noise is that?"), f" Just the {animal}, it {ok} every morning.", f" Just the {animal}, it {bad} every morning.",
              "factual", "animal_grid")


def social_grids(b: PairBuilder):
    for req, cond in [("stay up late", "finish your homework"), ("have a sweet", "eat your dinner"), ("play outside", "put your coat on"),
                      ("watch cartoons", "tidy your room"), ("skip my bath", "promise to have one tomorrow"), ("have a friend over", "ask their mum first")]:
        b.add(ctx(f"Mum, can I {req}?"), f" Only if you {cond}, sweetheart.", f" Only if you sign the form, sir.", "social", "parent_grid")
        b.add(ctx(f"Dad, please can I {req}?"), f" Alright, but first {cond}.", f" Alright, madam, but first show your ticket.", "social", "parent_grid")
    for item in ["bread", "stamps", "a newspaper", "apples", "a bus ticket", "aspirin"]:
        b.add(ctx(f"Hello, I would like {item}, please."), " Of course, that will be two euros.", " Of course, now go and brush your teeth.",
              "social", "shop_grid")
        b.add(ctx(f"Do you sell {item}?"), " Yes, just over there on the left.", " Yes, darling, now off to bed.", "social", "shop_grid")
    for subj in ["maths", "reading", "spelling", "drawing", "science", "history"]:
        b.add(ctx(f"Miss, I finished my {subj} exercise."), " Well done, now start the next one.", " Well done, now take your medicine.",
              "social", "teacher_grid")
        b.add(ctx(f"Sir, I do not understand the {subj} question."), " Come to the front and I will explain.", " Come to the front and pay at the till.",
              "social", "teacher_grid")


def temporal_grids(b: PairBuilder):
    for h in range(7, 18):
        b.add(ctx(f"It is {h} o'clock now, when is the meeting at {h + 2}?"), " In two hours.", " Two hours ago.", "temporal", "hour_grid")
    for h in range(9, 20):
        b.add(ctx(f"It is {h} o'clock, did the {h - 2} o'clock bus leave?"), " Yes, two hours ago.", " No, it leaves in two hours.", "temporal", "hour_grid")
    for s, nxt, wrong in [("winter", "spring", "autumn"), ("spring", "summer", "winter"), ("summer", "autumn", "spring"), ("autumn", "winter", "summer")]:
        b.add(ctx(f"{s.capitalize()} is over, what comes next?"), f" {nxt.capitalize()}, of course.", f" {wrong.capitalize()}, of course.", "temporal", "season_order")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for i, d in enumerate(days):
        n = 5 - i
        b.add(ctx(f"It is {d}; when is the weekend?"), f" In {n} day{'s' if n > 1 else ''}, on Saturday.", f" {n} day{'s' if n > 1 else ''} ago, on Saturday.", "temporal", "weekend_countdown")
    for m1, m2 in [("breakfast", "lunch"), ("lunch", "dinner"), ("breakfast", "dinner")]:
        b.add(ctx(f"We had {m2} at the usual time; when was {m1}?"), f" Earlier today, before {m2}.", f" Later today, after {m2}.", "temporal", "meal_order")
        b.add(ctx(f"Is it {m1} or {m2} time?"), f" {m1.capitalize()}, {m2} comes later.", f" {m1.capitalize()}, {m2} came earlier.", "temporal", "meal_order")


def social(b: PairBuilder):
    """Wrong register or role: the reply belongs to a different speaker role."""
    R = [
        # (A line with role cue, good B, bad B)
        ("Good morning, doctor, my throat hurts.", " Let me have a look, open wide please.", " Let me have a look, open your textbook please."),
        ("Doctor, is it serious?", " No, just rest and drink plenty of water.", " No, just sit down and finish your dinner."),
        ("Teacher, I do not understand this sum.", " Let's go through it together on the board.", " Let's go through it together at the till."),
        ("Miss, can I go to the toilet?", " Yes, but be quick, the lesson is starting.", " Yes, but be quick, the patient is waiting."),
        ("Mum, can I have a biscuit?", " After dinner, sweetheart, not before.", " After the invoice, sir, not before."),
        ("Dad, can you read me a story?", " Of course, pick one and get into bed.", " Of course, sir, here is the menu."),
        ("Excuse me, waiter, could we order?", " Certainly, what would you like?", " Certainly, please open your books at page ten."),
        ("Waiter, the soup is cold.", " I am sorry, I will bring a hot one.", " I am sorry, go to your room."),
        ("Driver, does this bus stop at the hospital?", " Yes, I will call it out for you.", " Yes, I will prescribe it for you."),
        ("Officer, I think I am lost.", " Where are you trying to get to?", " Where are you trying to sleep tonight, darling?"),
        ("Hello, I would like to buy some bread.", " Of course, white or brown?", " Of course, have you done your homework?"),
        ("Boss, can I take Friday off?", " Yes, if the report is finished.", " Yes, sweetheart, if you brush your teeth."),
        ("Grandma, can we bake a cake?", " Yes, dear, get the flour from the cupboard.", " Yes, sir, please sign here."),
        ("Nurse, when can I go home?", " The doctor will tell you this afternoon.", " The teacher will tell you this afternoon."),
        ("Mummy, I am scared of the dark.", " It is alright, I will leave the light on.", " That is not my department, madam."),
        ("Sir, your table is ready.", " Thank you, we will come through.", " Thank you, now go and tidy your room."),
        ("Hello, this is the plumber, where is the leak?", " Under the kitchen sink, come in.", " Under the kitchen sink, doctor, come in."),
        ("Could you check my ticket, please?", " Of course, that is fine, have a good trip.", " Of course, that is fine, say ah."),
        ("Mechanic, is the car ready?", " Almost, I am changing the oil.", " Almost, I am marking your exam."),
        ("Teacher, when is the test?", " On Friday, revise chapters one and two.", " On Friday, take two tablets a day."),
        ("Hello, I have an appointment with the dentist.", " Take a seat, he will call you shortly.", " Take a seat, the bus leaves shortly."),
        ("Can I help you find something?", " Yes, where are the eggs?", " Yes, where is my mummy?"),
        ("Welcome back, how was the holiday?", " Lovely, thanks, very relaxing.", " Lovely, thanks, that will be ten euros."),
        ("Daddy, I fell over and hurt my knee.", " Come here, let me see, we will clean it.", " Please hold, I will transfer you."),
        ("Good evening, table for two?", " Yes, please, by the window if possible.", " Yes, please, and can I go out to play?"),
    ]
    for a, g, bd in R:
        b.add(ctx(a), g, bd, "social", "role_swap", notes="bad reply fits a different role or register")
    # register: formal vs intimate, minimal swaps on the closing
    for a, g, bd in [
        ("Good morning, Mr Brown, how can I help you?", " Good morning, I would like to open an account.", " Morning, sweetie, give me a cuddle."),
        ("Hi love, I am home!", " Hi, dinner is almost ready.", " Good afternoon, sir, do you have a reservation?"),
        ("Dear customers, the shop closes in five minutes.", " Thank you, we are just paying now.", " Thank you, we are just going to bed now."),
        ("Night night, sleep tight.", " Night night, see you in the morning.", " Yes, sir, your receipt is in the bag."),
        ("Could you please hold the door, madam?", " Of course, go ahead.", " Of course, sweetie, go to sleep."),
        ("Happy birthday, darling!", " Thank you, what a lovely cake!", " Thank you, please take a number."),
        ("Ladies and gentlemen, please fasten your seatbelts.", " Thank you, we are ready.", " Thank you, mummy, we are ready."),
        ("Come on, kids, time for school!", " Coming, I just need my bag.", " Certainly, sir, the invoice is attached."),
        ("Thank you for calling, how may I help?", " Hello, I would like to change my appointment.", " Hello, pass me the salt, love."),
        ("Hey, want to come to the park?", " Sure, let's go after lunch.", " Please state your account number."),
    ]:
        b.add(ctx(a), g, bd, "social", "register_swap", notes="bad reply has the wrong register for the relationship")
    # role × request grid (systematic): who answers the door / phone
    roles = {"the doctor": "your prescription", "the teacher": "your homework", "the shopkeeper": "your change",
             "the waiter": "your order", "the bus driver": "your ticket", "the plumber": "your leaking tap",
             "the mechanic": "your car", "the librarian": "your library books"}
    names = list(roles)
    for i, r in enumerate(names):
        other = names[(i + 3) % len(names)]
        b.add(ctx(f"Who was that at the door?"), f" It was {r}, about {roles[r]}.", f" It was {r}, about {roles[other]}.",
              "social", "role_grid", notes="role and concern mismatch")
        b.add(ctx(f"What did {r} want?"), f" To talk about {roles[r]}.", f" To talk about {roles[other]}.",
              "social", "role_grid")
        b.add(ctx(f"Is {r} here yet?"), f" Yes, waiting with {roles[r]}.", f" Yes, waiting with {roles[other]}.",
              "social", "role_grid")
    # who-says-what
    for a, g, bd in [
        ("Who told you to go to bed?", " Mum did, it is late.", " The waiter did, it is late."),
        ("Who gave you the homework?", " The teacher did, this morning.", " The plumber did, this morning."),
        ("Who fixed the tap?", " The plumber did, this morning.", " The teacher did, this morning."),
        ("Who took your temperature?", " The nurse did, it is normal.", " The bus driver did, it is normal."),
        ("Who brought the menu?", " The waiter did, here it is.", " The doctor did, here it is."),
        ("Who checked the tickets?", " The driver did, at the door.", " The dentist did, at the door."),
        ("Who marked the test?", " The teacher did, I got a B.", " The shopkeeper did, I got a B."),
        ("Who served us at the bakery?", " The baker did, very kindly.", " The nurse did, very kindly."),
        ("Who tucked the baby in?", " Grandma did, very gently.", " The mechanic did, very gently."),
        ("Who is reading you a story tonight?", " Dad is, after my bath.", " The boss is, after my bath."),
    ]:
        b.add(ctx(a), g, bd, "social", "who_does_what", notes="the role named cannot plausibly do the action")


def temporal(b: PairBuilder):
    """Impossible or reversed sequences of everyday events."""
    chain = ["woke up", "had breakfast", "went to work", "had lunch", "came home", "had dinner", "went to bed"]
    for i in range(len(chain) - 1):
        e1, e2 = chain[i], chain[i + 1]
        b.add(ctx("What did you do today?"), f" I {e1} and then I {e2}.", f" I {e2} and then I {e1}.", "temporal", "day_chain")
        b.add(ctx(f"Had you already {e1} when you {e2}?"), f" Yes, I {e1} first.", f" No, I {e2} first.", "temporal", "day_chain_q")
    for i in range(len(chain) - 2):
        e1, e3 = chain[i], chain[i + 2]
        b.add(ctx("How did the day go?"), f" First I {e1}, later I {e3}.", f" First I {e3}, later I {e1}.", "temporal", "day_chain_skip")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i, d in enumerate(days):
        nxt, prv = days[(i + 1) % 7], days[(i - 1) % 7]
        wrong = days[(i + 3) % 7]
        b.add(ctx(f"Yesterday was {d}, so what is today?"), f" Today is {nxt}.", f" Today is {wrong}.", "temporal", "day_order")
        b.add(ctx(f"Tomorrow is {d}, so what is today?"), f" Today is {prv}.", f" Today is {wrong}.", "temporal", "day_order")
        b.add(ctx(f"Is it {nxt} already?"), f" Yes, {d} was yesterday.", f" Yes, {d} is tomorrow.", "temporal", "day_order_q")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    for i, m in enumerate(months):
        nxt, wrong = months[(i + 1) % 12], months[(i + 5) % 12]
        b.add(ctx(f"It is the end of {m}, what comes next?"), f" {nxt}, of course.", f" {wrong}, of course.", "temporal", "month_order")
    for a, g, bd in [
        ("Did you eat dinner yet?", " Yes, and then I washed the dishes.", " Yes, and then I had breakfast."),
        ("When did you buy the bread?", " This morning, before breakfast.", " Tomorrow, before breakfast."),
        ("Is the cake ready?", " Not yet, it has only been in the oven ten minutes.", " Not yet, we ate it an hour ago."),
        ("Has the bus gone?", " Yes, it left five minutes ago.", " Yes, it leaves in five minutes."),
        ("When will grandma arrive?", " In an hour, she is on the train now.", " An hour ago, she is on the train now."),
        ("Did you lock the door when you left?", " Yes, just after I closed it.", " Yes, just before I built it."),
        ("How old is the baby?", " Six months, she was born in spring.", " Sixty years, she was born in spring."),
        ("When did you finish school?", " Years ago, I work now.", " Next year, I retired last month."),
        ("Are the clothes dry?", " Yes, I hung them out this morning.", " Yes, I will wash them tomorrow."),
        ("Is it dark already?", " Yes, the sun set an hour ago.", " Yes, the sun rises in an hour."),
        ("Have you planted the seeds?", " Yes, in spring, and now they are flowering.", " Yes, in spring, after they flowered."),
        ("Did the alarm go off?", " Yes, at seven, and then I got up.", " Yes, at seven, before I went to bed at seven."),
        ("When do we open the presents?", " After dinner, once everyone is here.", " After dinner, before we cook it."),
        ("Is lunch over?", " Yes, we finished at two.", " Yes, we start at two."),
        ("When did you last see the keys?", " Last night, before bed.", " Tomorrow night, before bed."),
        ("Has the milk gone off?", " Yes, I bought it two weeks ago.", " Yes, I will buy it next week."),
        ("Was the shop open?", " Yes, I went this morning.", " Yes, I will go yesterday."),
        ("Did you catch the early train?", " Yes, the one at seven.", " Yes, the one at seven tonight."),
        ("How long have you lived here?", " Ten years, since the children were small.", " Ten years, since next summer."),
        ("When does the film end?", " At ten, it started at eight.", " At ten, it starts at eleven."),
        ("Is the washing done?", " Yes, the machine finished, so I hung it up.", " Yes, I hung it up, then the machine started."),
        ("When did it start raining?", " About an hour ago, the ground is wet.", " Tomorrow, the ground is wet."),
        ("Did the children go to school?", " Yes, this morning, they are back now.", " Yes, tonight, they are back now."),
        ("When do you retire?", " Next year, after forty years of work.", " Last century, after next year."),
        ("When is the meeting?", " At ten, so we have an hour to prepare.", " At ten, so we have an hour since it ended."),
        ("Have you boiled the kettle?", " Yes, so the tea is brewing now.", " Yes, so the tea was brewing before."),
        ("Did you brush your teeth?", " Yes, after breakfast.", " Yes, before I was born."),
        ("When did winter start?", " In December, a month ago.", " In December, next spring."),
        ("Is the bread baked?", " Yes, it came out of the oven ten minutes ago.", " Yes, it goes into the oven ten minutes ago."),
        ("Did you sleep well?", " Yes, I went to bed at ten and woke at seven.", " Yes, I woke at seven and then went to bed at ten this morning."),
    ]:
        b.add(ctx(a), g, bd, "temporal", "sequence", notes="bad reply implies an impossible order of events")


OPENERS = [
    "Good morning, did you sleep well?", "Hello, how are you today?", "What are you doing this afternoon?",
    "Shall we go for a walk after lunch?", "It is so cold today, isn't it?", "Have you had breakfast yet?",
    "What time do you finish work today?", "Can you help me carry the shopping?", "Where are you going?",
    "Is the bus late again?", "What would you like for dinner tonight?", "Did you hear the weather forecast?",
    "How was school today?", "Could you turn the heating up a bit?", "Are we still going to grandma's on Sunday?",
    "I cannot find my phone anywhere.", "The washing machine is making a strange noise.", "Do you want tea or coffee?",
    "Look, it is snowing!", "What day is it today?", "Can I borrow the car this evening?", "Who left the lights on?",
    "Is there anything good on television?", "The shop was closed when I got there.", "We need to leave in ten minutes.",
    "I think I left my umbrella on the bus.", "How long does it take to get to the station?", "Do you know a good bakery nearby?",
    "Happy birthday!", "Welcome home, how was the trip?", "Have you seen the cat?", "Can you pick up some milk on the way back?",
    "My bike has a flat tyre.", "It is too hot to sleep.", "Shall I open the window?", "What are your plans for the weekend?",
    "The children are already in bed.", "Sorry I am late, the traffic was terrible.", "Could you pass me the salt?",
    "Is this seat free?", "Excuse me, does this bus go to the hospital?", "I would like a loaf of bread, please.",
    "Good evening, do you have a table for two?", "Doctor, I have had a headache since Monday.", "Miss, I forgot my homework.",
    "Hi, I am your new neighbour.", "Can you recommend something from the menu?", "Where is the nearest pharmacy?",
    "How much is this jacket?", "Excuse me, what time does the library close?", "It looks like rain, doesn't it?",
    "Are you ready to order?", "I would like to book a haircut for Friday.", "The oven does not seem to be heating.",
    "What did you think of the film?", "I am so tired today.", "Are you coming to the match on Saturday?",
    "Let's invite the neighbours for a barbecue.", "Could you water the plants while I am away?", "The heating bill was very high this month.",
    "Can we get a dog?", "The bath is ready for the kids.", "Do you remember where we parked?", "The train leaves at half past eight.",
    "I made soup, do you want some?", "It gets dark so early now.", "How do you get to work?", "Would you like some more cake?",
    "My alarm did not go off this morning.", "Shall we take the bike or walk?", "It is your turn to do the dishes.",
    "What should we get dad for his birthday?", "I saw a rainbow on the way home.", "Can you lend me a pen?",
    "How many people are coming on Saturday?", "The garden needs mowing.", "Is the pool open in winter?",
    "Did you lock the back door?", "The baby finally fell asleep.", "We are out of coffee.",
    "Do you want to watch a film tonight?", "I have a dentist appointment at three.", "Could you read the recipe out to me?",
    "The bus was so crowded this morning.", "What time is the school play?", "Let's make pancakes on Sunday.",
    "Has the post come yet?", "I am thinking of painting the bedroom.", "Is it going to be sunny tomorrow?",
    "Good night, see you in the morning.", "Grandpa is coming for lunch.", "Can you teach me to ride a bike?",
    "The kettle has just boiled.", "Do you have change for the bus?", "What is your favourite season?",
    "We should leave before the rush hour.", "The neighbours' dog kept me awake.", "I passed my driving test!",
    "Could you check if the shop is open on Sundays?", "Why is the floor wet?", "Are the shops busy on Saturday mornings?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="suites/dialogue_v1")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    b = PairBuilder("dlg", seed=args.seed, author=AUTHOR)
    pragmatic(b); factual(b); factual_grids(b); social(b); social_grids(b); temporal(b); temporal_grids(b)
    pairs = sorted(b.items(), key=lambda i: i.id)
    by_type: dict[str, list[Item]] = {}
    for it in pairs:
        by_type.setdefault(it.category, []).append(it)
    for k, v in sorted(by_type.items()):
        print(f"pairs/{k:10s} {len(v)}")

    prompts = [Item(id=f"dlg-open-{i:03d}", kind=Kind.PROGRAM, prompt=ctx(a), checker="benchsmith.checks.dialogue:score",
                    category="opener", max_tokens=48, provenance=Provenance(method="hand", source="OPENERS", author=AUTHOR))
               for i, a in enumerate(OPENERS)]
    print(f"prompts           {len(prompts)}")

    tasks = [Task(name=f"pairs_{k}", file=f"pairs_{k}.jsonl", kind=Kind.PAIR_LOGLIK, scoring={"normalize": "mean"},
                  description=f"{k} violation: length-normalised logprob(good) > logprob(bad)") for k in sorted(by_type)]
    tasks.append(Task(name="openers", file="prompts.jsonl", kind=Kind.PROGRAM, scoring={"temperature": 0.0},
                      description="QUALITATIVE: lenient format/alternation/on-topic check of a generated reply"))
    items = {f"pairs_{k}": v for k, v in by_type.items()}
    items["openers"] = prompts
    meta = SuiteMeta(name="dialogue", version="1.0",
                     description="Track 3 — everyday dialogue coherence (out-of-distribution: no dialogue source in training).",
                     tasks=tasks, meta={"violation_types": sorted(by_type), "qualitative_tasks": ["openers"]})
    save_suite(args.out, meta, items)
    print("total", len(pairs) + len(prompts), "->", args.out)


if __name__ == "__main__":
    main()

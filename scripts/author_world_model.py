"""Author the Track 1 'world model' suite: closed-set cloze probes scored by log-likelihood.

Conventions (see suites/world_model_v1/README.md):
  * prompts end at the blank, with NO trailing space;
  * every option carries a leading space (" Tuesday"), so options are interchangeable
    under a byte-level BPE;
  * distractors are same word-class and drawn from the same vocabulary as the answer;
  * answers cycle over the whole vocabulary so a frequency prior scores ≈ chance.

Run:  python scripts/author_world_model.py [--out suites/world_model_v1] [--seed 1000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchsmith.authoring.templates import Builder
from benchsmith.items import Kind, SuiteMeta, Task, save_suite

SEED = 1000

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
DAYS_IN = {"January": 31, "February": 28, "March": 31, "April": 30, "May": 31, "June": 30, "July": 31,
           "August": 31, "September": 30, "October": 31, "November": 30, "December": 31}
SEASON_OF = {"December": "winter", "January": "winter", "February": "winter",
             "March": "spring", "April": "spring", "May": "spring",
             "June": "summer", "July": "summer", "August": "summer",
             "September": "autumn", "October": "autumn", "November": "autumn"}
SEASONS = ["spring", "summer", "autumn", "winter"]
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
            "tenth", "eleventh", "twelfth"]


def sp(x) -> str:
    return " " + str(x)


# ----------------------------------------------------------------------------- calendar
def calendar(b: Builder):
    n = len(DAYS)
    for i, d in enumerate(DAYS):
        nxt, prv, nxt2, prv2 = DAYS[(i + 1) % n], DAYS[(i - 1) % n], DAYS[(i + 2) % n], DAYS[(i - 2) % n]
        for tpl, ans in [
            (f"The day after {d} is", nxt), (f"The day that comes after {d} is", nxt),
            (f"If today is {d}, tomorrow is", nxt), (f"{d} is followed by", nxt),
            (f"The day before {d} is", prv), (f"If today is {d}, yesterday was", prv),
            (f"{d} comes right after", prv), (f"The day that comes before {d} is", prv),
            (f"Two days after {d} is", nxt2), (f"If today is {d}, the day after tomorrow is", nxt2),
            (f"Two days before {d} is", prv2), (f"If today is {d}, the day before yesterday was", prv2),
            (f"The day between {prv} and {nxt} is", d), (f"The day right after {prv} and right before {nxt} is", d),
            (f"A week after {d} it is again", d), (f"Seven days after {d} is", d),
        ]:
            b.add(tpl, sp(ans), [sp(x) for x in b.pick(DAYS, 3, {ans, d})], "day_adjacency", "calendar")
        wk = "weekend" if d in ("Saturday", "Sunday") else "weekday"
        # balance weekday/weekend: weekend facts get more phrasings
        phr = [(f"{d} is a", wk), (f"For most people, {d} is a", wk), (f"On the calendar, {d} counts as a", wk)]
        if wk == "weekend":
            phr += [(f"People usually do not go to work on {d}, because it is part of the", "weekend"),
                    (f"Schools are closed on {d}, which is part of the", "weekend"),
                    (f"{d} belongs to the", "weekend"), (f"A day off for most workers, {d} is a", "weekend day"),
                    (f"Most offices are closed on {d}, a", "weekend day"), (f"Many families rest on {d}, a", "weekend day"),
                    (f"{d} is one of the two days of the", "weekend")]
        for tpl, ans in phr:
            if ans in ("weekday", "weekend day"):
                b.add(tpl, sp(ans), [sp("weekend day" if ans == "weekday" else "weekday")], "weekday_membership", "calendar")
            else:
                b.add(tpl, sp(ans), [sp("working week")], "weekday_membership", "calendar")
    for i, m in enumerate(MONTHS):
        nxt, prv = MONTHS[(i + 1) % 12], MONTHS[(i - 1) % 12]
        for tpl, ans in [
            (f"The month after {m} is", nxt), (f"{m} is followed by the month of", nxt),
            (f"The month before {m} is", prv), (f"{m} comes right after", prv),
            (f"The month between {prv} and {nxt} is", m), (f"{ORDINALS[i].capitalize()} month of the year:", m),
            (f"The {ORDINALS[i]} month of the year is", m),
        ]:
            b.add(tpl, sp(ans), [sp(x) for x in b.pick(MONTHS, 3, {ans, m})], "month_order", "calendar")
        b.add(f"{m} is the", sp(ORDINALS[i]), [sp(x) for x in b.pick(ORDINALS, 3, ORDINALS[i])],
              "month_ordinal", "calendar", notes="prompt continues: '... month of the year'")
        dn = DAYS_IN[m]
        for tpl in [f"{m} has", f"The number of days in {m} is", f"A normal (non-leap) year's {m} has"]:
            b.add(tpl, sp(dn), [sp(x) for x in (28, 30, 31) if x != dn], "days_per_month", "calendar",
                  notes="continuation '... days'")
        s = SEASON_OF[m]
        for tpl in [f"In the northern hemisphere, {m} is in", f"In Europe, {m} is a month of",
                    f"North of the equator, {m} belongs to the season called"]:
            b.add(tpl, sp(s), [sp(x) for x in SEASONS if x != s], "month_season", "calendar")
    for tpl, ans, pool in [
        ("The number of days in a week is", "seven", ["five", "six", "ten", "twelve"]),
        ("A week has", "seven", ["five", "six", "ten", "twelve"]),
        ("The number of months in a year is", "twelve", ["seven", "ten", "eleven", "thirty"]),
        ("A year has", "twelve", ["seven", "ten", "eleven", "thirty"]),
        ("The number of hours in a day is", "twenty-four", ["twelve", "twenty", "thirty", "sixty"]),
        ("The number of minutes in an hour is", "sixty", ["twelve", "twenty-four", "thirty", "one hundred"]),
        ("The number of seconds in a minute is", "sixty", ["twelve", "twenty-four", "thirty", "one hundred"]),
        ("The number of days in a year is about", "365", ["100", "300", "400", "1000"]),
        ("The number of weekend days in a week is", "two", ["one", "three", "five"]),
        ("The number of working days in a normal week is", "five", ["two", "three", "seven"]),
        ("The number of weeks in a year is about", "52", ["12", "30", "100"]),
        ("The number of seasons in a year is", "four", ["two", "three", "six"]),
        ("The first day of the working week is usually", "Monday", ["Friday", "Saturday", "Sunday"]),
        ("The last working day of a normal week is", "Friday", ["Monday", "Sunday", "Wednesday"]),
        ("The first month of the year is", "January", ["December", "March", "June"]),
        ("The last month of the year is", "December", ["January", "November", "August"]),
        ("The shortest month of the year is", "February", ["January", "April", "December"]),
        ("In a leap year, February has", "29", ["28", "30", "31"]),
        ("A decade is a period of", "ten", ["five", "twelve", "one hundred"]),
        ("A century is a period of", "one hundred", ["ten", "fifty", "one thousand"]),
        ("The middle of the working week is", "Wednesday", ["Monday", "Friday", "Sunday"]),
        ("Noon is at", "12:00", ["6:00", "9:00", "18:00"]),
        ("Midnight is at", "00:00", ["06:00", "12:00", "18:00"]),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in pool], "calendar_counts", "calendar")


# ----------------------------------------------------------------------------- seasons & weather
def seasons_weather(b: Builder):
    season_facts = {
        "summer": ["The hottest season of the year is", "The warmest time of the year is",
                   "The season with the longest days is", "People go to the beach most often in",
                   "The season when schools have their long holiday is", "Ice cream sells best in",
                   "The season with the most daylight is", "The sun sets latest in",
                   "The season when people wear shorts and sandals is", "The warmest season is",
                   "Fans and air conditioners are used most in", "The season right after spring is",
                   "The season right before autumn is", "The hottest months of the year belong to"],
        "winter": ["The coldest season of the year is", "The season when it snows is",
                   "The season with the shortest days is", "The sun sets earliest in",
                   "People wear heavy coats and gloves in", "Lakes can freeze over in",
                   "The season with the least daylight is", "Heating is used most in",
                   "The season when days are short and nights are long is", "The coldest months belong to",
                   "People build snowmen in", "The season right after autumn is",
                   "The season right before spring is", "Frost on the windows is typical of"],
        "spring": ["The season when flowers start to bloom is", "The season when trees grow new leaves is",
                   "The season that comes after winter is", "Snow melts and days grow longer in",
                   "Baby birds hatch most often in", "The season right before summer is",
                   "The season when the weather turns from cold to warm is", "Plants start to grow again in",
                   "Lambs are usually born in", "Many people plant their gardens in",
                   "The season between winter and summer is", "Days get longer and warmer in"],
        "autumn": ["The season when leaves turn red and yellow is", "The season when leaves fall from the trees is",
                   "The season that comes after summer is", "Apples and pumpkins are harvested in",
                   "The season right before winter is", "The weather turns from warm to cold in",
                   "The season between summer and winter is", "Days get shorter and cooler in",
                   "Farmers bring in the harvest in", "People rake leaves in",
                   "Birds fly south at the end of", "The season also called fall is"],
    }
    for s, tpls in season_facts.items():
        for t in tpls:
            b.add(t, sp(s), [sp(x) for x in SEASONS if x != s], "season_identity", "seasons_weather")
    # hot/cold, long/short, earlier/later — binary and balanced by construction
    for tpl, ans, other in [
        ("In summer the weather is usually", "hot", "cold"), ("In winter the weather is usually", "cold", "hot"),
        ("A typical July day in Europe feels", "warm", "freezing"), ("A typical January day in Europe feels", "cold", "hot"),
        ("Compared with summer, in winter it gets dark", "earlier", "later"),
        ("Compared with winter, in summer it gets dark", "later", "earlier"),
        ("Compared with winter, in summer the sun rises", "earlier", "later"),
        ("Compared with summer, in winter the sun rises", "later", "earlier"),
        ("In summer the days are", "long", "short"), ("In winter the days are", "short", "long"),
        ("In winter the nights are", "long", "short"), ("In summer the nights are", "short", "long"),
        ("In summer, water in a lake is", "warm", "frozen"), ("In winter, water in a pond may be", "frozen", "boiling"),
        ("Snow falls when the air is", "cold", "hot"), ("Ice melts when the weather gets", "warm", "cold"),
        ("In summer people dress", "lightly", "warmly"), ("In winter people dress", "warmly", "lightly"),
        ("When the temperature is below zero, water", "freezes", "boils"),
        ("When the weather is hot, ice cream", "melts", "freezes"),
        ("In summer the heating at home is usually", "off", "on"), ("In winter the heating at home is usually", "on", "off"),
        ("In hot weather people drink", "more", "less"), ("In cold weather people go swimming outdoors", "less", "more"),
        ("The temperature at noon is usually", "higher", "lower"), ("The temperature at night is usually", "lower", "higher"),
        ("In summer the temperature is", "higher", "lower"), ("In winter the temperature is", "lower", "higher"),
        ("When it is sunny, the sky is", "blue", "grey"), ("When it is about to rain, the sky is", "grey", "blue"),
        ("In the middle of summer, the afternoon is usually the", "hottest", "coldest"),
        ("In the middle of winter, the night is usually the", "coldest", "hottest"),
        ("After rain, the ground is", "wet", "dry"), ("After many sunny days, the ground is", "dry", "wet"),
    ]:
        b.add(tpl, sp(ans), [sp(other)], "weather_binary", "seasons_weather")
    # typical temperatures (Celsius)
    temps = ["-10", "0", "10", "20", "30"]
    for tpl, ans in [
        ("On a hot summer day the temperature is about", "30"), ("On a cold winter day the temperature can be about", "-10"),
        ("Water freezes at", "0"), ("A mild spring day is about", "20"), ("A cool autumn morning is about", "10"),
        ("A warm summer evening is about", "20"), ("Snow stays on the ground when it is below", "0"),
        ("A pleasant room temperature is about", "20"), ("A freezer keeps food at about", "-10"),
        ("On a hot beach day the air is about", "30"), ("Ice starts to melt above", "0"),
        ("A chilly late-autumn day is about", "10"), ("In a heatwave the temperature can reach", "30"),
        ("A very cold night in the mountains can be about", "-10"), ("A fresh early-spring day is about", "10"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(temps, 3, ans)], "temperature", "seasons_weather",
              notes="continuation '... degrees Celsius'")
    # weather-activity coupling
    wx = ["sunny", "rainy", "snowy", "windy", "foggy", "stormy"]
    for tpl, ans in [
        ("People carry an umbrella when the weather is", "rainy"), ("People fly kites when the weather is", "windy"),
        ("Children build snowmen when the weather is", "snowy"), ("People wear sunglasses when the weather is", "sunny"),
        ("Drivers turn on their lights in daytime when it is", "foggy"), ("People stay indoors and avoid the sea when it is", "stormy"),
        ("Raincoats and boots are worn when it is", "rainy"), ("Skiing is possible when the weather has been", "snowy"),
        ("Hats can blow away when it is", "windy"), ("It is hard to see far ahead when it is", "foggy"),
        ("People sunbathe when it is", "sunny"), ("Thunder and lightning come with weather that is", "stormy"),
        ("Sailing boats move fastest when it is", "windy"), ("Streets get slippery with white flakes when it is", "snowy"),
        ("Picnics are most pleasant when it is", "sunny"), ("Puddles form on the road when it is", "rainy"),
        ("Ferries may be cancelled when the sea is", "stormy"), ("Airports delay flights when it is very", "foggy"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(wx, 3, ans)], "weather_activity", "seasons_weather")
    cloth = ["a coat", "shorts", "a swimsuit", "gloves", "a raincoat", "a scarf", "sandals", "boots"]
    for tpl, ans in [
        ("In cold weather people wear", "a coat"), ("In hot weather people wear", "shorts"),
        ("At the beach people wear", "a swimsuit"), ("To keep their hands warm people wear", "gloves"),
        ("When it rains people wear", "a raincoat"), ("To keep their neck warm people wear", "a scarf"),
        ("In summer many people wear", "sandals"), ("In snow people wear", "boots"),
        ("On a freezing morning you put on", "a coat"), ("For a swim in the sea you need", "a swimsuit"),
        ("In a downpour the most useful thing to wear is", "a raincoat"), ("On a hot day the most comfortable shoes are", "sandals"),
        ("In deep snow the most useful shoes are", "boots"), ("When it is windy and cold, around your neck you wrap", "a scarf"),
        ("Skiers keep their fingers warm with", "gloves"), ("On a hot afternoon people prefer to wear", "shorts"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(cloth, 3, ans)], "clothing", "seasons_weather")
    sky = ["rain", "snow", "hail", "fog", "wind", "sunshine"]
    for tpl, ans in [
        ("Water that falls from clouds as drops is", "rain"), ("White flakes that fall in winter are", "snow"),
        ("Small balls of ice falling from the sky are", "hail"), ("A cloud at ground level that hides the view is", "fog"),
        ("Moving air that bends the trees is", "wind"), ("Bright light and warmth from a clear sky is", "sunshine"),
        ("Umbrellas protect you from", "rain"), ("Snowploughs clear the roads after heavy", "snow"),
        ("Car roofs can be dented by", "hail"), ("Lighthouses help ships in thick", "fog"),
        ("Windmills are turned by the", "wind"), ("Sunburn comes from too much", "sunshine"),
        ("Puddles are left behind by", "rain"), ("Sledges are used after", "snow"),
        ("Kites need plenty of", "wind"), ("Plants grow best with water and", "sunshine"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(sky, 3, ans)], "weather_words", "seasons_weather")
    for tpl, ans, other in [
        ("In spring, the temperature compared with winter is", "higher", "lower"),
        ("In autumn, the temperature compared with summer is", "lower", "higher"),
        ("From winter to summer, the days get", "longer", "shorter"), ("From summer to winter, the days get", "shorter", "longer"),
        ("In spring, the snow", "melts", "piles up"), ("In autumn, the leaves", "fall", "sprout"),
        ("In spring, trees", "bloom", "lose their leaves"), ("In winter, many trees are", "bare", "in full leaf"),
        ("The hottest part of the day is usually the", "afternoon", "early morning"),
        ("The coldest part of the day is usually just before", "dawn", "noon"),
        ("Near the equator the weather is", "hot", "freezing"), ("Near the poles the weather is", "freezing", "hot"),
        ("High in the mountains it is", "colder", "warmer"), ("Down by the sea in summer it is", "warmer", "colder"),
        ("In a desert, rain is", "rare", "constant"), ("In a rainforest, rain is", "frequent", "rare"),
        ("In summer the sun is", "high", "low"), ("In winter the sun is", "low", "high"),
        ("On a cloudy day the sun is", "hidden", "bright"), ("On a clear day the stars at night are", "visible", "hidden"),
        ("In summer, lawns need watering", "more", "less"), ("In winter, plants grow", "less", "more"),
    ]:
        b.add(tpl, sp(ans), [sp(other)], "weather_binary", "seasons_weather")
    for m in MONTHS:
        s = SEASON_OF[m]
        hot = s == "summer"
        cold = s == "winter"
        if hot or cold:
            b.add(f"In Europe, the weather in {m} is usually", sp("hot" if hot else "cold"),
                  [sp("cold" if hot else "hot")], "month_weather", "seasons_weather")
            b.add(f"In Europe, the days in {m} are", sp("long" if hot else "short"),
                  [sp("short" if hot else "long")], "month_weather", "seasons_weather")
            b.add(f"In Europe, the nights in {m} are", sp("short" if hot else "long"),
                  [sp("long" if hot else "short")], "month_weather", "seasons_weather")
        else:
            b.add(f"In Europe, the weather in {m} is usually", sp("mild"), [sp("freezing"), sp("scorching")],
                  "month_weather", "seasons_weather")


# ----------------------------------------------------------------------------- time conventions
def time_conventions(b: Builder):
    parts = ["morning", "afternoon", "evening", "night"]
    for tpl, ans in [
        ("People usually eat breakfast in the", "morning"), ("People usually eat lunch in the early", "afternoon"),
        ("People usually eat dinner in the", "evening"), ("People usually sleep at", "night"),
        ("Most people wake up in the", "morning"), ("Most people go to bed at", "night"),
        ("Children go to school in the", "morning"), ("Children come home from school in the", "afternoon"),
        ("Office workers usually start work in the", "morning"), ("Office workers usually finish work in the late", "afternoon"),
        ("Families often watch television together in the", "evening"), ("It is dark outside at", "night"),
        ("The sun rises in the", "morning"), ("The sun sets in the", "evening"),
        ("People brush their teeth before bed at", "night"), ("A nap after lunch happens in the", "afternoon"),
        ("The first meal of the day is eaten in the", "morning"), ("The last meal of the day is eaten in the", "evening"),
        ("Shops usually open in the", "morning"), ("Most people are asleep at", "night"),
        ("Coffee is most often drunk in the", "morning"), ("Schools finish in the", "afternoon"),
        ("After dinner, people relax in the", "evening"), ("Streets are quietest at", "night"),
        ("People take a shower before work in the", "morning"), ("Homework is usually done in the", "afternoon"),
        ("Bedtime stories are read at", "night"), ("The evening news is broadcast in the", "evening"),
        ("A lunch break comes at the start of the", "afternoon"), ("Owls hunt at", "night"),
        ("A commute to work usually happens in the", "morning"), ("A commute home usually happens in the late", "afternoon"),
        ("Most restaurants are busiest for dinner in the", "evening"), ("The moon is visible at", "night"),
        ("People say 'good morning' in the", "morning"), ("People say 'good night' at", "night"),
        ("Afternoon tea is served in the", "afternoon"), ("People say 'good evening' in the", "evening"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in parts if x != ans], "part_of_day", "time_conventions")
    for tpl, ans in [
        ("Birds sing loudest in the early", "morning"), ("Dew is found on the grass in the", "morning"),
        ("Shops are busiest after work in the late", "afternoon"), ("Street lights come on in the", "evening"),
        ("The postman usually delivers letters in the", "morning"), ("Children play outside after school in the", "afternoon"),
        ("People cook dinner in the early", "evening"), ("Hospitals are quietest at", "night"),
        ("The morning rush hour is in the", "morning"), ("The evening rush hour is in the late", "afternoon"),
        ("Bakers start work very early in the", "morning"), ("A night shift is worked at", "night"),
        ("People read the newspaper with breakfast in the", "morning"), ("Sports matches on television are often shown in the", "evening"),
        ("The school bell for the first lesson rings in the", "morning"), ("The school bell for the last lesson rings in the", "afternoon"),
        ("People set their alarm to ring in the", "morning"),
        ("Babies are put to bed in the", "evening"), ("The siesta in hot countries is taken in the", "afternoon"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in parts if x != ans], "part_of_day", "time_conventions")
    clock = ["7:00", "9:00", "12:30", "15:00", "17:30", "19:30", "23:00"]
    for tpl, ans in [
        ("Breakfast is usually eaten at about", "7:00"), ("Lunch is usually eaten at about", "12:30"),
        ("Dinner is usually eaten at about", "19:30"), ("Most people are asleep by", "23:00"),
        ("Most offices open at about", "9:00"), ("Most offices close at about", "17:30"),
        ("School lessons usually end at about", "15:00"), ("An alarm clock for work typically rings at about", "7:00"),
        ("A typical bedtime for adults is around", "23:00"), ("The evening meal is served at around", "19:30"),
        ("The midday meal is served at around", "12:30"), ("Children get out of school at around", "15:00"),
        ("The working day usually begins at about", "9:00"), ("The working day usually ends at about", "17:30"),
        ("Shops in town usually open at about", "9:00"), ("Shops in town usually close at about", "17:30"),
        ("Most people wake up at about", "7:00"), ("Most people have their lunch break at", "12:30"),
        ("Late-night television starts at about", "23:00"), ("Afternoon school activities finish at about", "15:00"),
        ("The evening news is usually on at about", "19:30"), ("People leave for work at about", "7:00"),
        ("Morning lessons at school start at about", "9:00"), ("The last customers leave the office building at about", "17:30"),
        ("A restaurant's dinner service is busiest at about", "19:30"), ("The streets are quiet and dark at", "23:00"),
        ("A school canteen serves lunch at about", "12:30"), ("After-school clubs begin at about", "15:00"),
        ("Breakfast television starts at about", "7:00"), ("Banks open their doors at about", "9:00"),
        ("Most people are back home from work by", "17:30"), ("Families sit down for the evening meal at about", "19:30"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(clock, 3, ans)], "clock_time", "time_conventions")
    meals = ["breakfast", "lunch", "dinner"]
    for tpl, ans in [
        ("The first meal of the day is", "breakfast"), ("The meal eaten around noon is", "lunch"),
        ("The meal eaten in the evening is", "dinner"), ("The meal eaten right after waking up is", "breakfast"),
        ("The meal eaten in the middle of the working day is", "lunch"), ("The last meal of the day is", "dinner"),
        ("Cereal and toast are typical for", "breakfast"), ("A sandwich at the office is a typical", "lunch"),
        ("A family sits down together in the evening for", "dinner"), ("The meal before school is", "breakfast"),
        ("The meal children eat at school is", "lunch"), ("The meal after work is", "dinner"),
        ("Coffee and juice usually go with", "breakfast"), ("A one-hour break at work is for", "lunch"),
        ("The meal eaten at about seven in the evening is", "dinner"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in meals if x != ans], "meal_names", "time_conventions")
    hours = ["one", "two", "five", "eight", "twelve"]
    for tpl, ans in [
        ("Adults usually sleep about", "eight"), ("A full-time working day is about", "eight"),
        ("A school day lasts about", "five"), ("A lunch break lasts about", "one"),
        ("A film at the cinema lasts about", "two"), ("A football match lasts about", "two"),
        
        ("A night's sleep for a child is about", "twelve"),
        ("A commute by car usually takes about", "one"), ("A school lesson lasts about", "one"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(hours, 3, ans)], "durations", "time_conventions",
              notes="continuation '... hours' or '... days' (working week)")
    freq = ["every day", "once a week", "once a month", "once a year"]
    for tpl, ans in [
        ("People brush their teeth", "every day"), ("People eat breakfast", "every day"),
        ("Many families do a big grocery shop", "once a week"), ("Most people get paid a salary", "once a month"),
        ("People celebrate their birthday", "once a year"), ("People celebrate New Year", "once a year"),
        ("Office workers commute", "every day"), ("Rubbish is collected", "once a week"),
        ("The rent is paid", "once a month"), ("Christmas comes", "once a year"),
        ("Most people sleep", "every day"), ("Many people go to the market", "once a week"),
        ("Electricity bills usually arrive", "once a month"), ("Schools have a long summer holiday", "once a year"),
        ("People have dinner", "every day"), ("Many people visit their grandparents", "once a week"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in freq if x != ans], "frequency", "time_conventions")
    for tpl, ans, other in [
        ("Breakfast comes", "before", "after"), ("Dinner comes", "after", "before"),
        ("People wake up", "before", "after"), ("People go to bed", "after", "before"),
        ("Lunch is eaten", "after", "before"), ("The commute to work happens", "before", "after"),
        ("Children do their homework", "after", "before"), ("The alarm clock rings", "before", "after"),
        ("Sunrise comes", "before", "after"), ("Sunset comes", "after", "before"),
        ("Monday's work shift starts", "before", "after"), ("Saturday comes", "after", "before"),
    ]:
        notes = {"Breakfast comes": "… lunch", "Dinner comes": "… lunch", "People wake up": "… breakfast",
                 "People go to bed": "… dinner", "Lunch is eaten": "… breakfast", "The commute to work happens": "… the working day",
                 "Children do their homework": "… school", "The alarm clock rings": "… people get up",
                 "Sunrise comes": "… noon", "Sunset comes": "… noon", "Monday's work shift starts": "… Friday's",
                 "Saturday comes": "… Friday"}[tpl]
        b.add(tpl, sp(ans), [sp(other)], "ordering", "time_conventions", notes=notes)


# ----------------------------------------------------------------------------- physical / social
def physical_social(b: Builder):
    rooms = ["kitchen", "bedroom", "bathroom", "living room", "garage", "garden"]
    for tpl, ans in [
        ("You cook in the", "kitchen"), ("You sleep in the", "bedroom"), ("You take a shower in the", "bathroom"),
        ("You watch television in the", "living room"), ("You park the car in the", "garage"),
        ("You grow flowers in the", "garden"), ("The fridge is in the", "kitchen"), ("The bed is in the", "bedroom"),
        ("The toilet is in the", "bathroom"), ("The sofa is in the", "living room"), ("Tools and bikes are kept in the", "garage"),
        ("You mow the lawn in the", "garden"), ("Dishes are washed in the", "kitchen"), ("You get dressed in the", "bedroom"),
        ("You brush your teeth in the", "bathroom"), ("Guests sit and chat in the", "living room"),
        ("The car is repaired in the", "garage"), ("Vegetables are planted in the", "garden"),
        ("Meals are prepared in the", "kitchen"), ("The wardrobe is in the", "bedroom"), ("The bathtub is in the", "bathroom"),
        ("The television is in the", "living room"), ("A barbecue is held in the", "garden"),
        ("The oven is in the", "kitchen"), ("Pyjamas are kept in the", "bedroom"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(rooms, 3, ans)], "room_activity", "physical_social")
    appl = {
        "fridge": ["keeps food cold", "keeps milk fresh"], "oven": ["bakes bread", "roasts a chicken"],
        "washing machine": ["cleans clothes", "washes dirty laundry"], "kettle": ["boils water", "heats water for tea"],
        "vacuum cleaner": ["cleans the floor", "sucks up dust"], "lamp": ["gives light", "lights up a room"],
        "dishwasher": ["washes dishes", "cleans plates and cups"], "heater": ["warms a room", "keeps a house warm"],
        "freezer": ["keeps food frozen", "stores ice cream"], "toaster": ["browns bread", "makes toast"],
        "iron": ["smooths clothes", "removes creases from shirts"], "alarm clock": ["wakes people up", "rings in the morning"],
    }
    names = list(appl)
    for a, fns in appl.items():
        for fn in fns:
            b.add(f"The appliance that {fn} is the", sp(a), [sp(x) for x in b.pick(names, 3, a)], "appliance_function", "physical_social")
            b.add(f"At home, the thing that {fn} is the", sp(a), [sp(x) for x in b.pick(names, 3, a)],
                  "appliance_function", "physical_social")
    acts = ["cold", "hot", "clean", "dry", "bright", "warm"]
    for tpl, ans in [
        ("A fridge keeps food", "cold"), ("An oven makes food", "hot"), ("A washing machine makes clothes", "clean"),
        ("A tumble dryer makes clothes", "dry"), ("A lamp makes a room", "bright"), ("A heater makes a room", "warm"),
        ("A freezer keeps things very", "cold"), ("A kettle makes water", "hot"), ("Soap makes hands", "clean"),
        ("A towel makes you", "dry"), ("Sunlight makes a room", "bright"), ("A blanket keeps you", "warm"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(acts, 3, ans)], "appliance_effect", "physical_social")
    # transport
    modes = ["on foot", "by bicycle", "by car", "by train", "by plane"]
    for tpl, ans in [
        ("The slowest way to travel is", "on foot"), ("The fastest way to cross an ocean is", "by plane"),
        ("To go to another continent people travel", "by plane"), ("To reach the next room you go", "on foot"),
        ("Children often get to a nearby school", "on foot"), ("A trip of ten kilometres without an engine is done", "by bicycle"),
        ("A family trip to another city on the motorway is made", "by car"), ("Commuters between two cities without driving go", "by train"),
        ("The cheapest way to travel a short distance is", "on foot"), ("A trip that needs an airport is", "by plane"),
        ("A vehicle with two wheels and pedals takes you", "by bicycle"), ("Travelling on rails means going", "by train"),
        ("Parking is needed when you travel", "by car"), ("A ticket and a platform are needed to travel", "by train"),
        ("A journey of five thousand kilometres in a day is made", "by plane"), ("To walk the dog you go", "on foot"),
        ("A helmet is worn when travelling", "by bicycle"), ("A driving licence is needed to travel", "by car"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(modes, 3, ans)], "transport_mode", "physical_social")
    dur = ["minutes", "hours", "days", "weeks"]
    for tpl, ans in [
        ("Walking to another city takes", "days"), ("Walking to the corner shop takes", "minutes"),
        ("Driving to a city two hundred kilometres away takes", "hours"), ("Flying to another continent takes", "hours"),
        ("Walking across a whole country takes", "weeks"), ("Crossing the street takes", "minutes"),
        ("A train trip between two big cities takes", "hours"), ("Boiling an egg takes", "minutes"),
        ("Cycling to the next village takes", "minutes"), ("Sailing across an ocean takes", "weeks"),
        ("A flight between neighbouring countries takes", "hours"), ("Walking fifty kilometres takes about a", "day"),
        ("Driving from one end of a large country to the other takes", "days"), ("A bus ride across town takes", "minutes"),
    ]:
        pool = dur + ["day"]
        b.add(tpl, sp(ans), [sp(x) for x in b.pick([d for d in pool if d != ("days" if ans == "day" else "day")], 3, ans)],
              "travel_duration", "physical_social")
    speed = [("a car", "a bicycle"), ("a bicycle", "a person walking"), ("a plane", "a train"), ("a train", "a bus"),
             ("a running person", "a walking person"), ("a motorbike", "a bicycle"), ("a train", "a car"), ("a plane", "a car")]
    for fast, slow in speed:
        b.add(f"Which is faster: {fast} or {slow}? The faster one is", sp(fast), [sp(slow)], "speed_compare", "physical_social")
        b.add(f"Which is slower: {fast} or {slow}? The slower one is", sp(slow), [sp(fast)], "speed_compare", "physical_social")
    dist = ["50 metres", "5 kilometres", "500 kilometres"]
    for tpl, ans in [
        ("A car wash that is very close, just a short walk away, is about", "50 metres"),
        ("A bike ride to the next village is about", "5 kilometres"), ("A drive to a city far away is about", "500 kilometres"),
        ("The distance across a large supermarket car park is about", "50 metres"),
        ("A distance that takes an hour to walk is about", "5 kilometres"), ("A distance that takes a day's drive is about", "500 kilometres"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in dist if x != ans], "distance_scale", "physical_social")
    for tpl, ans, other in [
        ("A car wash fifty metres away: you should", "walk", "drive"), ("A shop five hundred kilometres away: you should", "drive", "walk"),
        ("To get to the house next door you", "walk", "drive"), ("To get to a city across the country you", "drive", "walk"),
        ("For a fifty-metre trip it makes more sense to", "walk", "drive"), ("For a three-hundred-kilometre trip it makes more sense to", "drive", "walk"),
        ("To reach the end of your own street you", "walk", "drive"), ("To reach a town two hours away you", "drive", "walk"),
    ]:
        b.add(tpl, sp(ans), [sp(other)], "walk_or_drive", "physical_social")
    # household roles / occupations
    jobs = {"teacher": "teaches children at a school", "doctor": "treats sick people", "cook": "prepares meals in a restaurant",
            "bus driver": "drives people around town", "farmer": "grows crops and keeps animals", "nurse": "cares for patients in a hospital",
            "plumber": "fixes pipes and taps", "electrician": "fixes wires and lights", "baker": "bakes bread and cakes",
            "mechanic": "repairs cars", "shop assistant": "serves customers in a shop", "office worker": "works at a desk from nine to five",
            "student": "goes to classes and studies", "retired person": "no longer goes to work"}
    jn = list(jobs)
    for j, d in jobs.items():
        b.add(f"Someone who {d} is a", sp(j), [sp(x) for x in b.pick(jn, 3, j)], "occupation", "physical_social")
        b.add(f"The person who {d} is called a", sp(j), [sp(x) for x in b.pick(jn, 3, j)], "occupation", "physical_social")
    for tpl, ans, other in [
        ("An office worker spends weekdays", "at the office", "at home"), ("A retired person spends weekdays", "at home", "at the office"),
        ("A remote worker does their job", "from home", "from a factory"), ("A child on a school day is", "at school", "at the office"),
        ("An office worker commutes", "on weekdays", "on weekends"), ("A retired person has a commute", "never", "every day"),
        ("A school child has lessons", "on weekdays", "on weekends"), ("A shop assistant often works", "on Saturdays", "at night"),
        ("A night nurse sleeps", "during the day", "at night"), ("A farmer starts work", "early in the morning", "late at night"),
        ("A student spends the morning", "in class", "at the office"), ("A remote worker's commute takes", "no time", "two hours"),
    ]:
        b.add(tpl, sp(ans), [sp(other)], "household_roles", "physical_social")
    who = ["adults", "children", "babies", "elderly people"]
    for tpl, ans in [
        ("Those who go to work to earn money are usually", "adults"), ("Those who go to primary school are", "children"),
        ("Those who sleep most of the day and drink milk are", "babies"), ("Those who are retired are usually", "elderly people"),
        ("Those who drive cars are", "adults"), ("Those who play in the school playground are", "children"),
        ("Those who cannot walk or talk yet are", "babies"), ("Grandparents are usually", "elderly people"),
        ("Those who pay the household bills are the", "adults"), ("Those who get homework are the", "children"),
        ("Those who sleep in a cot are", "babies"), ("Those most likely to use a walking stick are", "elderly people"),
    ]:
        b.add(tpl, sp(ans), [sp(x) for x in b.pick(who, 3, ans)], "household_members", "physical_social")


# ----------------------------------------------------------------------------- general knowledge
def general_knowledge(b: Builder):
    caps = {"France": "Paris", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "the United Kingdom": "London",
            "Japan": "Tokyo", "China": "Beijing", "Russia": "Moscow", "Egypt": "Cairo", "Greece": "Athens",
            "Portugal": "Lisbon", "Austria": "Vienna", "Poland": "Warsaw", "Ireland": "Dublin", "Norway": "Oslo",
            "Sweden": "Stockholm", "Finland": "Helsinki", "Denmark": "Copenhagen", "the Netherlands": "Amsterdam",
            "Belgium": "Brussels", "Turkey": "Ankara", "India": "New Delhi", "Canada": "Ottawa", "Australia": "Canberra",
            "Brazil": "Brasília", "Argentina": "Buenos Aires", "Mexico": "Mexico City", "the United States": "Washington",
            "Kenya": "Nairobi", "South Korea": "Seoul", "Thailand": "Bangkok", "Hungary": "Budapest",
            "the Czech Republic": "Prague", "Switzerland": "Bern", "Peru": "Lima", "Cuba": "Havana"}
    cities = list(caps.values())
    for c, cap in caps.items():
        for tpl in [f"The capital of {c} is", f"The capital city of {c} is called"]:
            b.add(tpl, sp(cap), [sp(x) for x in b.pick(cities, 3, cap)], "capitals", "general_knowledge")
    countries = list(caps)
    for c, cap in caps.items():
        b.add(f"{cap} is the capital of", sp(c), [sp(x) for x in b.pick(countries, 3, c)], "capitals", "general_knowledge")
    classes = ["mammal", "bird", "fish", "reptile", "insect", "amphibian"]
    animals = {"mammal": ["dog", "cat", "cow", "whale", "horse", "elephant", "bat", "lion"],
               "bird": ["eagle", "sparrow", "penguin", "owl", "parrot", "duck", "chicken", "swan"],
               "fish": ["salmon", "shark", "trout", "goldfish", "tuna", "cod", "sardine", "eel"],
               "reptile": ["snake", "lizard", "crocodile", "turtle", "gecko", "chameleon", "iguana", "alligator"],
               "insect": ["ant", "bee", "butterfly", "beetle", "mosquito", "fly", "grasshopper", "ladybird"],
               "amphibian": ["frog", "toad", "salamander", "newt", "axolotl", "tree frog", "bullfrog", "caecilian"]}
    for cl, ans in animals.items():
        for a in ans:
            b.add(f"A {a} is a kind of", sp(cl), [sp(x) for x in b.pick(classes, 3, cl)], "animal_class", "general_knowledge")
    facts = [
        ("The largest animal on Earth is the", "blue whale", ["elephant", "giraffe", "shark"]),
        ("The tallest animal on Earth is the", "giraffe", ["elephant", "horse", "camel"]),
        ("The fastest land animal is the", "cheetah", ["horse", "lion", "wolf"]),
        ("The largest land animal is the", "elephant", ["rhino", "hippo", "bear"]),
        ("Birds are covered in", "feathers", ["fur", "scales", "shells"]),
        ("Fish breathe using", "gills", ["lungs", "wings", "fins"]),
        ("Fish live in", "water", ["trees", "deserts", "caves"]),
        ("Bees make", "honey", ["milk", "silk", "wool"]),
        ("Cows give us", "milk", ["honey", "eggs", "silk"]),
        ("Chickens lay", "eggs", ["seeds", "fruit", "milk"]),
        ("Sheep give us", "wool", ["silk", "honey", "eggs"]),
        ("A young dog is called a", "puppy", ["kitten", "calf", "lamb"]),
        ("A young cat is called a", "kitten", ["puppy", "foal", "chick"]),
        ("A young cow is called a", "calf", ["lamb", "cub", "puppy"]),
        ("A young sheep is called a", "lamb", ["calf", "kid", "chick"]),
        ("Spiders have", "eight", ["two", "four", "six"]),
        ("Insects have", "six", ["two", "four", "eight"]),
        ("Water boils at", "100", ["0", "50", "200"]),
        ("Water freezes at", "0", ["10", "50", "100"]),
        ("Ice is frozen", "water", ["air", "sand", "oil"]),
        ("Steam is water in the form of a", "gas", ["solid", "metal", "rock"]),
        ("The Sun is a", "star", ["planet", "moon", "comet"]),
        ("The Earth is a", "planet", ["star", "moon", "comet"]),
        ("The Moon goes around the", "Earth", ["Sun", "Mars", "stars"]),
        ("The Earth goes around the", "Sun", ["Moon", "Mars", "stars"]),
        ("The planet we live on is called", "Earth", ["Mars", "Venus", "Jupiter"]),
        ("The largest planet in the solar system is", "Jupiter", ["Earth", "Mars", "Venus"]),
        ("The planet closest to the Sun is", "Mercury", ["Earth", "Mars", "Jupiter"]),
        ("The red planet is", "Mars", ["Venus", "Earth", "Saturn"]),
        ("The planet with big rings is", "Saturn", ["Mars", "Venus", "Mercury"]),
        ("The number of planets in the solar system is", "eight", ["five", "ten", "twelve"]),
        ("Plants need sunlight, water and", "soil", ["meat", "plastic", "metal"]),
        ("Plants make their food using", "sunlight", ["meat", "coal", "salt"]),
        ("The green colour of leaves comes from", "chlorophyll", ["iron", "sugar", "salt"]),
        ("Trees take in carbon dioxide and give out", "oxygen", ["nitrogen", "helium", "smoke"]),
        ("Humans breathe in", "oxygen", ["helium", "carbon", "hydrogen"]),
        ("The gas humans breathe out is", "carbon dioxide", ["oxygen", "helium", "neon"]),
        ("The organ that pumps blood is the", "heart", ["brain", "lung", "stomach"]),
        ("The organ used for thinking is the", "brain", ["heart", "liver", "kidney"]),
        ("The organs used for breathing are the", "lungs", ["kidneys", "bones", "ears"]),
        ("The organ that digests food is the", "stomach", ["heart", "brain", "lung"]),
        ("Humans see with their", "eyes", ["ears", "hands", "feet"]),
        ("Humans hear with their", "ears", ["eyes", "nose", "teeth"]),
        ("Humans smell with their", "nose", ["ears", "eyes", "feet"]),
        ("The hard parts that hold up the body are the", "bones", ["muscles", "veins", "nerves"]),
        ("An adult human has", "206", ["20", "100", "1000"]),
        ("The number of teeth in an adult mouth is about", "32", ["10", "20", "64"]),
        ("Blood is", "red", ["blue", "green", "yellow"]),
        ("The sky on a clear day looks", "blue", ["green", "red", "black"]),
        ("Grass is usually", "green", ["blue", "red", "purple"]),
        ("Snow is", "white", ["black", "green", "red"]),
        ("Ripe bananas are", "yellow", ["blue", "purple", "black"]),
        ("A ripe tomato is", "red", ["blue", "black", "white"]),
        ("Coal is", "black", ["white", "pink", "green"]),
        ("Mixing blue and yellow paint gives", "green", ["red", "purple", "orange"]),
        ("Mixing red and white paint gives", "pink", ["green", "blue", "black"]),
        ("Mixing red and yellow paint gives", "orange", ["green", "purple", "blue"]),
        ("A dozen is", "twelve", ["six", "ten", "twenty"]),
        ("Half of ten is", "five", ["two", "four", "six"]),
        ("Two plus two equals", "four", ["three", "five", "six"]),
        ("Ten minus three equals", "seven", ["six", "eight", "thirteen"]),
        ("Three times three equals", "nine", ["six", "eight", "twelve"]),
        ("A triangle has", "three", ["two", "four", "five"]),
        ("A square has", "four", ["three", "five", "six"]),
        ("The largest ocean is the", "Pacific", ["Atlantic", "Indian", "Arctic"]),
        ("The longest river in Africa is the", "Nile", ["Amazon", "Danube", "Thames"]),
        ("The highest mountain in the world is", "Everest", ["Kilimanjaro", "Mont Blanc", "Fuji"]),
        ("The largest desert that is hot is the", "Sahara", ["Gobi", "Atacama", "Kalahari"]),
        ("The continent where Italy is found is", "Europe", ["Asia", "Africa", "Australia"]),
        ("The continent where Japan is found is", "Asia", ["Europe", "Africa", "America"]),
        ("The continent where Egypt is found is", "Africa", ["Europe", "Asia", "Australia"]),
        ("The continent where Brazil is found is", "South America", ["Europe", "Asia", "Africa"]),
        ("The continent where Canada is found is", "North America", ["Europe", "Asia", "Africa"]),
        ("The frozen continent at the South Pole is", "Antarctica", ["Africa", "Asia", "Europe"]),
        ("The number of continents is", "seven", ["four", "five", "ten"]),
        ("The ocean between Europe and America is the", "Atlantic", ["Pacific", "Indian", "Arctic"]),
        ("The language spoken in France is", "French", ["German", "Spanish", "Italian"]),
        ("The language spoken in Germany is", "German", ["French", "Spanish", "Italian"]),
        ("The language spoken in Spain is", "Spanish", ["German", "French", "Italian"]),
        ("The language spoken in Italy is", "Italian", ["German", "Spanish", "French"]),
        ("The language spoken in Brazil is", "Portuguese", ["Spanish", "French", "Italian"]),
        ("The language spoken in Japan is", "Japanese", ["Chinese", "Korean", "Thai"]),
        ("The currency used in most of Europe is the", "euro", ["dollar", "pound", "yen"]),
        ("The currency of the United States is the", "dollar", ["euro", "pound", "yen"]),
        ("The currency of Japan is the", "yen", ["euro", "dollar", "pound"]),
        ("The currency of the United Kingdom is the", "pound", ["euro", "dollar", "yen"]),
        ("The metal that magnets attract is", "iron", ["gold", "wood", "glass"]),
        ("A material that floats on water is", "wood", ["iron", "stone", "gold"]),
        ("A material that sinks in water is", "stone", ["wood", "cork", "foam"]),
        ("Glass is made from", "sand", ["wood", "oil", "milk"]),
        ("Paper is made from", "wood", ["sand", "oil", "stone"]),
        ("Bread is made from", "flour", ["sand", "wool", "stone"]),
        ("Cheese is made from", "milk", ["flour", "sand", "oil"]),
        ("Wine is made from", "grapes", ["apples", "wheat", "milk"]),
        ("Something that is very hot will cause a", "burn", ["bruise", "cut", "cold"]),
        ("Things fall to the ground because of", "gravity", ["magnets", "wind", "light"]),
        ("Light travels faster than", "sound", ["nothing", "thought", "time"]),
        ("Thunder is the sound made by", "lightning", ["rain", "wind", "snow"]),
        ("A rainbow appears when there is sun and", "rain", ["snow", "fog", "wind"]),
        ("Clouds are made of tiny drops of", "water", ["oil", "sand", "smoke"]),
        ("Rivers flow into the", "sea", ["sky", "mountains", "desert"]),
        ("A doctor works in a", "hospital", ["school", "farm", "garage"]),
        ("A teacher works in a", "school", ["hospital", "farm", "garage"]),
        ("Books are borrowed from a", "library", ["bakery", "garage", "hospital"]),
        ("Bread is bought at a", "bakery", ["library", "garage", "pharmacy"]),
        ("Medicine is bought at a", "pharmacy", ["bakery", "library", "garage"]),
        ("Trains stop at a", "station", ["harbour", "airport", "garage"]),
        ("Planes take off from an", "airport", ["harbour", "station", "office"]),
        ("Ships dock in a", "harbour", ["station", "airport", "garage"]),
        ("The number of letters in the English alphabet is", "26", ["10", "20", "30"]),
        ("The first letter of the alphabet is", "A", ["B", "C", "Z"]),
        ("The last letter of the alphabet is", "Z", ["A", "X", "Y"]),
        ("Shakespeare was a famous", "writer", ["painter", "scientist", "king"]),
        ("Mozart was a famous", "composer", ["painter", "scientist", "king"]),
        ("Einstein was a famous", "scientist", ["painter", "composer", "king"]),
        ("Picasso was a famous", "painter", ["writer", "scientist", "king"]),
        ("The first people to walk on the Moon were from the", "United States", ["United Kingdom", "France", "China"]),
        ("Dinosaurs lived", "millions", ["hundreds", "thousands", "dozens"]),
        ("The Romans built a famous amphitheatre in Rome called the", "Colosseum", ["Parthenon", "Louvre", "Kremlin"]),
        ("The famous tower in Paris is the", "Eiffel Tower", ["Big Ben", "Colosseum", "Parthenon"]),
        ("The famous clock tower in London is", "Big Ben", ["the Eiffel Tower", "the Colosseum", "the Parthenon"]),
        ("The ancient pyramids are found in", "Egypt", ["Italy", "Greece", "Japan"]),
    ]
    for tpl, ans, distract in facts:
        b.add(tpl, sp(ans), [sp(x) for x in distract], "basic_facts", "general_knowledge")


PREFIXES = ["Usually,", "In everyday life,", "As everyone knows,", "Generally speaking,", "For most people,"]


def fill_with_paraphrases(b: Builder, target: int):
    """Top a task up to `target` items with prefixed paraphrases of existing items.

    Paraphrases are tagged `paraphrase` and keep the original option set, so they
    add prompt-context variety but NOT new facts; README documents this. Answers
    stay balanced because we cycle through items in a seeded random order.
    """
    base = b.items()
    rng = b.rng
    order = base[:]
    rng.shuffle(order)
    k = 0
    while len(b.items()) < target and k < len(order) * len(PREFIXES):
        src = order[k % len(order)]
        prefix = PREFIXES[(k // len(order)) % len(PREFIXES)]
        k += 1
        prompt = prefix + " " + src.prompt[0].lower() + src.prompt[1:]
        b.add(prompt, src.answer, [o for o in src.options if o != src.answer],
              src.provenance.source + "+prefix", src.category, tags=["paraphrase"], notes=src.notes)


TRACKS = [
    ("calendar", calendar, "Calendar structure: days, months, ordering, counts, seasons of months"),
    ("seasons_weather", seasons_weather, "Seasons & weather: season identity, temperatures, daylight, weather–activity coupling"),
    ("time_conventions", time_conventions, "Everyday time conventions: meals, work/school hours, sleep, frequencies"),
    ("physical_social", physical_social, "Everyday physical/social facts: rooms, appliances, transport, roles"),
    ("general_knowledge", general_knowledge, "General knowledge control: capitals, animals, basic science (Simple-Wikipedia-like)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="suites/world_model_v1")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-rebalance", action="store_true")
    ap.add_argument("--min-items", type=int, default=200)
    args = ap.parse_args()

    tasks, items = [], {}
    for name, fn, desc in TRACKS:
        b = Builder(task=name, seed=args.seed, author="template:scripts/author_world_model.py")
        fn(b)
        n0 = len(b.items())
        if len(b.items()) < args.min_items:
            fill_with_paraphrases(b, int(args.min_items * 1.15))
        dropped = 0 if args.no_rebalance else b.rebalance(max_ratio=1.5)
        its = sorted(b.items(), key=lambda i: i.id)
        tasks.append(Task(name=name, file=f"{name}.jsonl", kind=Kind.CHOICE_LOGLIK, description=desc,
                          scoring={"normalize": "mean"}))
        items[name] = its
        print(f"{name:18s} {len(its):4d} items  (authored {n0}, dropped {dropped} for balance)")
    meta = SuiteMeta(
        name="world_model", version="1.0",
        description="Track 1 — world-model likelihood probes for tiny models (closed-set cloze, per-token-mean logprob).",
        tasks=tasks,
    )
    save_suite(args.out, meta, items)
    print("total", sum(len(v) for v in items.values()), "->", args.out)


if __name__ == "__main__":
    main()

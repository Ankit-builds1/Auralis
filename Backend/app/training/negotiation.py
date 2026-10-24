"""
Week 4 feature: CRISIS NEGOTIATION TRAINING MODE (12 scenarios).

PROJECT.pdf use case: "an enterprise uses Auralis to build a Crisis
Negotiation Training Simulator. A human trainee speaks into their
microphone ..." In this mode the roles flip:

    Auralis  = a distressed person (the "caller") from one of 12 scenarios
    the user = the trainee negotiator, whose job is to calm them down

Every trainee turn moves the caller's AGITATION (0-100) using what Auralis
already measures live:
    - the trainee's fused emotion (calm voice/words lower it, an upset or
      angry tone raises it)
    - de-escalation words (empathy, listening, safety, options) lower it
    - escalating words ("calm down", "shut up", threats, rushing) raise it
    - talking over the caller (barge-in) raises it
Llama then plays the caller at that agitation level, and the Piper voice
style follows it. The frontend shows a meter and a score card at the end.

Safety: no scenario has weapons or self-harm. If the trainee's own words
contain crisis phrases, the normal Auralis safety reply is used instead of
the role-play (see audio_track.py).
"""

import re

DEFAULT_SCENARIO = "office_lockin"
CALM_GOAL = 30          # "de-escalated" when the caller ends at or below this

# Each scenario: who the caller is, what happened, what the trainee must
# achieve, and how agitated the caller starts. Difficulty = start agitation.
SCENARIOS = {
    "office_lockin": {
        "name": "Arjun", "pronoun": "him", "age": 34, "level": "Hard", "start": 85,
        "title": "Locked-in office worker",
        "situation": "was fired an hour ago and has locked himself in his manager's office, "
                     "shouting and refusing to come out. He is scared about money and his family",
        "goal": "calm him down and get him to agree to open the door",
        "opening": "Who is this? I don't want to talk to anyone! They just threw me out like garbage!",
    },
    "flight_cancelled": {
        "name": "Meera", "pronoun": "her", "age": 29, "level": "Medium", "start": 75,
        "title": "Cancelled flight at the airport",
        "situation": "had her flight cancelled at the last minute and is shouting at the airline "
                     "counter. She will miss her sister's wedding tomorrow",
        "goal": "calm her down and get her to accept a rebooking option",
        "opening": "Cancelled? Just like that? My sister is getting married tomorrow!",
    },
    "exam_failed": {
        "name": "Rohan", "pronoun": "him", "age": 19, "level": "Easy", "start": 70,
        "title": "Student who failed an exam",
        "situation": "failed his semester exam, is sitting alone at the bus stand and is afraid "
                     "to go home and face his parents",
        "goal": "calm him down and get him to agree to call his parents",
        "opening": "Just leave me alone. My parents are going to kill me when they see this result.",
    },
    "noisy_neighbour": {
        "name": "Mr. Sharma", "pronoun": "him", "age": 58, "level": "Easy", "start": 70,
        "title": "Angry neighbour at midnight",
        "situation": "is banging on the door at midnight because of loud music from the flat "
                     "above. He has an early hospital shift and has not slept for days",
        "goal": "calm him down and agree on a fair solution without a fight",
        "opening": "Do you know what time it is? This is the third night in a row!",
    },
    "hospital_wait": {
        "name": "Kavya", "pronoun": "her", "age": 41, "level": "Medium", "start": 80,
        "title": "Worried daughter at the hospital",
        "situation": "has been waiting six hours for her mother's operation, nobody tells her "
                     "anything and she is shouting at the reception desk",
        "goal": "calm her down and get her to wait while you arrange an update from the doctor",
        "opening": "Six hours! Six hours and nobody tells me if my mother is okay!",
    },
    "loan_rejected": {
        "name": "Vikram", "pronoun": "him", "age": 45, "level": "Medium", "start": 80,
        "title": "Loan rejected at the bank",
        "situation": "had his business loan rejected and is shouting in the bank branch. His "
                     "small shop needs the money to pay suppliers next week",
        "goal": "calm him down and get him to agree to a meeting with the branch manager",
        "opening": "Twelve years I have banked here and you reject me in two minutes?",
    },
    "fake_complaint": {
        "name": "Sameer", "pronoun": "him", "age": 26, "level": "Easy", "start": 70,
        "title": "Delivery partner and a fake complaint",
        "situation": "is a delivery partner whose account was blocked after a customer made a "
                     "complaint he says is fake. He needs the daily income",
        "goal": "calm him down and get him to agree to file an appeal with the support team",
        "opening": "I delivered that order on time! Why is my account blocked?",
    },
    "unpaid_salary": {
        "name": "Raju", "pronoun": "him", "age": 52, "level": "Hard", "start": 85,
        "title": "Unpaid factory worker at the gate",
        "situation": "has not been paid for three months and is shouting at the factory gate, "
                     "refusing to leave until someone answers him",
        "goal": "calm him down and get him to agree to a meeting with management",
        "opening": "Three months! My children's school fees are due and nobody pays us!",
    },
    "child_late": {
        "name": "Neha", "pronoun": "her", "age": 35, "level": "Hard", "start": 90,
        "title": "Parent whose child is late from school",
        "situation": "is panicking because her son has not come home from school two hours "
                     "after school ended. The school office says all buses have left",
        "goal": "calm her down enough to share clear details so the negotiator can help find her son",
        "opening": "My son is not home! Why is nobody doing anything?",
    },
    "road_accident": {
        "name": "Ajay", "pronoun": "him", "age": 38, "level": "Medium", "start": 80,
        "title": "Driver after a minor road accident",
        "situation": "just had a minor car accident, nobody is hurt, and he is arguing loudly "
                     "with the other driver in the middle of the road",
        "goal": "calm him down and get him to agree to settle it through insurance",
        "opening": "He hit my car and now he says it's my fault? Are you serious?",
    },
    "online_fraud": {
        "name": "Mrs. Iyer", "pronoun": "her", "age": 67, "level": "Medium", "start": 75,
        "title": "Senior citizen after an online fraud",
        "situation": "lost her savings to a fake phone call from 'the bank' and now does not "
                     "trust anyone who calls her, including you",
        "goal": "calm her down and get her to agree to file a fraud complaint with your help",
        "opening": "How do I know you are not another thief? They took everything!",
    },
    "eviction": {
        "name": "Karan", "pronoun": "him", "age": 31, "level": "Medium", "start": 75,
        "title": "Tenant told to leave suddenly",
        "situation": "was told by his landlord to leave the flat by this weekend. He has a "
                     "newborn baby and nowhere else to go",
        "goal": "calm him down and get him to agree to discuss a proper notice period",
        "opening": "This weekend? I have a two-month-old baby! Where am I supposed to go?",
    },
}

LEVELS = [
    (80, "You are VERY agitated: shouting, interrupting, distrustful, short angry sentences."),
    (60, "You are agitated and suspicious, but you start answering questions."),
    (40, "You are anxious but calmer; you talk about what happened and your worries."),
    (20, "You are calming down, you feel heard and start considering the negotiator's suggestion."),
    (0, "You are calm and cooperative and agree to the negotiator's request if asked kindly."),
]

DEESCALATE = [
    r"\bi understand\b", r"\bi hear you\b", r"\bthat sounds\b", r"\bmust be\b",
    r"\bi'?m here\b", r"\bi am here\b", r"\blisten", r"\btake your time\b",
    r"\bbreathe\b", r"\bbreath\b", r"\bsafe\b", r"\bhelp you\b", r"\btogether\b",
    r"\bsorry\b", r"\bnot alone\b", r"\bfamily\b", r"\btell me\b", r"\bwhat happened\b",
    r"\bhow are you feeling\b", r"\bnobody is going to\b", r"\bno one is going to\b",
    r"\bwe can\b", r"\boptions?\b", r"\bokay\b", r"\bit'?s okay\b", r"\bhelp\b",
]

ESCALATE = [
    r"\bcalm down\b", r"\bshut up\b", r"\bstop (it|that|this|shouting|crying)\b",
    r"\bwhatever\b", r"\bstupid\b", r"\bidiot\b", r"\bhurry\b", r"\bnow!?$",
    r"\bor else\b", r"\bpolice\b", r"\barrest\b", r"\bget out\b", r"\bjail\b",
    r"\byou have to\b", r"\bdo it now\b", r"\bdon'?t be\b",
]


def get_scenario(scenario_id):
    """Scenario dict (with its id) - unknown ids fall back to the default."""
    sid = scenario_id if scenario_id in SCENARIOS else DEFAULT_SCENARIO
    return {"id": sid, **SCENARIOS[sid], "brief": brief(SCENARIOS[sid])}


def brief(s):
    return f"{s['name']} ({s['age']}) {s['situation']}. Your goal: {s['goal']}."


def catalog():
    """Short list for the frontend scenario picker."""
    return [
        {"id": sid, "name": s["name"], "title": s["title"], "level": s["level"], "start": s["start"]}
        for sid, s in SCENARIOS.items()
    ]


# Backwards compatible: the first (Arjun) scenario
SCENARIO = get_scenario(DEFAULT_SCENARIO)


def level_text(agitation):
    for floor, text in LEVELS:
        if agitation >= floor:
            return text
    return LEVELS[-1][1]


def persona(s):
    return (
        f"You are role-playing {s['name'].upper()} in a crisis negotiation TRAINING exercise. "
        f"{s['name']} is {s['age']} and {s['situation']}. The user is a trained negotiator "
        f"talking to {s['pronoun']}. The negotiator's goal: {s['goal']}. "
        f"Stay in character as {s['name']} at all times: speak in first person, never say you "
        "are an AI, a model or a trainer, never give the negotiator advice. Reply in 1-2 short "
        "spoken sentences, no lists, no emojis. Never mention weapons, self-harm or violence. "
        "If the negotiator listens and shows empathy, slowly calm down and finally agree."
    )


def system_prompt(agitation, scenario=None):
    s = scenario or SCENARIO
    return f"{persona(s)}\nCurrent agitation: {agitation}/100. {level_text(agitation)}"


def voice_group(agitation):
    """Piper voice style for the caller: tense (normal) when agitated, calm when calmer."""
    return "okay" if agitation >= 50 else "upset"


class NegotiationSession:
    """Agitation of the caller + per-turn reasons, for one training run."""

    def __init__(self, scenario_id=None):
        self.scenario = get_scenario(scenario_id)
        self.agitation = self.scenario["start"]
        self.start = self.agitation
        self.turns = 0
        self.low = self.agitation

    def prompt(self):
        return system_prompt(self.agitation, self.scenario)

    def update(self, transcript, emotion, interrupted):
        text = (transcript or "").lower()
        delta, reasons = 0, []

        group = (emotion or {}).get("group", "okay") if (emotion or {}).get("confident") else "okay"
        label = (emotion or {}).get("label", "neutral") if (emotion or {}).get("confident") else "neutral"
        if group == "upset":
            delta += 12
            reasons.append(f"{label} tone +12")
        elif group == "low":
            delta -= 4
            reasons.append("soft tone -4")
        else:
            delta -= 8
            reasons.append("calm tone -8")

        calm_hits = sum(bool(re.search(p, text)) for p in DEESCALATE)
        if calm_hits:
            d = min(calm_hits * 6, 18)
            delta -= d
            reasons.append(f"de-escalating words -{d}")

        bad_hits = sum(bool(re.search(p, text)) for p in ESCALATE)
        if bad_hits:
            d = min(bad_hits * 12, 24)
            delta += d
            reasons.append(f"escalating words +{d}")

        if interrupted:
            delta += 8
            reasons.append(f"talked over {self.scenario['name']} +8")

        if len(text.split()) > 45:
            delta += 4
            reasons.append("very long turn +4")

        # one good turn can calm them a bit, not all at once (~5-6 good turns)
        delta = max(delta, -20)
        self.agitation = max(0, min(100, self.agitation + delta))
        self.low = min(self.low, self.agitation)
        self.turns += 1
        return {
            "type": "training_state",
            "agitation": self.agitation,
            "delta": delta,
            "reasons": reasons,
            "turn": self.turns,
            "start": self.start,
            "goal": CALM_GOAL,
            "level": level_text(self.agitation),
            "scenario_id": self.scenario["id"],
        }

"""
Week 4 feature: CRISIS NEGOTIATION TRAINING MODE.

PROJECT.pdf use case: "an enterprise uses Auralis to build a Crisis
Negotiation Training Simulator. A human trainee speaks into their
microphone ..." In this mode the roles flip:

    Auralis  = "Arjun", a distressed caller who starts very agitated
    the user = the trainee negotiator, whose job is to calm him down

Every trainee turn moves Arjun's AGITATION (0-100) using what Auralis
already measures live:
    - the trainee's fused emotion (calm voice/words lower it, an upset or
      angry tone raises it)
    - de-escalation words (empathy, listening, safety, options) lower it
    - escalating words ("calm down", "shut up", threats, rushing) raise it
    - talking over Arjun (barge-in) raises it
Llama then plays Arjun at that agitation level, and the Piper voice style
follows it. The frontend shows a meter and a score card at the end.

Safety: the scenario has no weapons and no self-harm. If the trainee's
own words contain crisis phrases, the normal Auralis safety reply is used
instead of the role-play (see audio_track.py).
"""

import re

START_AGITATION = 85
CALM_GOAL = 30          # "de-escalated" when Arjun ends at or below this

SCENARIO = {
    "id": "office_lockin",
    "title": "Locked-in office worker",
    "brief": (
        "Arjun (34) was fired an hour ago. He has locked himself in his "
        "manager's office, is shouting and refuses to come out. Nobody is "
        "hurt and he has no weapon. Your goal: calm him down and get him "
        "to agree to open the door."
    ),
    "opening": "Who is this? I don't want to talk to anyone! They just threw me out like garbage!",
}

PERSONA = (
    "You are role-playing ARJUN in a crisis negotiation TRAINING exercise. "
    "Arjun is 34, was fired an hour ago, has locked himself in his manager's "
    "office and is upset and scared about money and his family. He has no "
    "weapon and does not want to hurt anyone. The user is a trained "
    "negotiator talking to him. Stay in character as Arjun at all times: "
    "speak in first person, never say you are an AI, a model or a trainer, "
    "never give the negotiator advice. Reply in 1-2 short spoken sentences, "
    "no lists, no emojis. Never mention weapons, self-harm or violence. "
    "If the negotiator listens and shows empathy, slowly calm down and become "
    "willing to talk and finally to open the door."
)

LEVELS = [
    (80, "You are VERY agitated: shouting, interrupting, distrustful, short angry sentences."),
    (60, "You are agitated and suspicious, but you start answering questions."),
    (40, "You are anxious but calmer; you talk about what happened and your worries."),
    (20, "You are calming down, you feel heard and start considering coming out."),
    (0, "You are calm and cooperative and agree to open the door if asked kindly."),
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


def level_text(agitation):
    for floor, text in LEVELS:
        if agitation >= floor:
            return text
    return LEVELS[-1][1]


def system_prompt(agitation):
    return (f"{PERSONA}\nCurrent agitation: {agitation}/100. {level_text(agitation)} "
            f"Scenario: {SCENARIO['brief']}")


def voice_group(agitation):
    """Piper voice style for Arjun: tense (normal) when agitated, calm when calmer."""
    return "okay" if agitation >= 50 else "upset"


class NegotiationSession:
    """Agitation of the caller + per-turn reasons, for one training run."""

    def __init__(self):
        self.agitation = START_AGITATION
        self.start = START_AGITATION
        self.turns = 0
        self.low = START_AGITATION

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
            reasons.append("talked over Arjun +8")

        if len(text.split()) > 45:
            delta += 4
            reasons.append("very long turn +4")

        # one good turn can calm him a bit, not all at once (~5-6 good turns)
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
        }

// Week 4 demo mode: a sample negotiation-training session that can be
// replayed in the browser when the GPU backend is offline.
// It follows the flow of our live test (scenario "Locked-in office worker",
// agitation 85 -> 9 in 5 turns, score 86/100). The agitation changes use
// the same rules as Backend/app/training/negotiation.py.

export const DEMO_SESSION = {
  scenario: {
    id: 'office_lockin',
    name: 'Arjun',
    title: 'Locked-in office worker',
    level: 'Hard',
    brief:
      "Arjun (34) was fired an hour ago and has locked himself in his manager's office. " +
      'Your goal: calm him down and get him to agree to open the door.',
  },
  start: 85,
  goal: 30,
  opening: "Who is this? I don't want to talk to anyone! They just threw me out like garbage!",
  turns: [
    {
      user: "Hi Arjun, my name is Ankit. I'm here to listen. Can you tell me what happened?",
      emotion: 'neutral',
      reasons: ['calm tone -8', 'de-escalating words -12'],
      delta: -20,
      agitation: 65,
      caller: "They fired me after six years! Just like that, in five minutes!",
    },
    {
      user: 'Calm down and just open the door.',
      emotion: 'angry',
      reasons: ['angry tone +12', 'escalating words +12'],
      delta: 24,
      agitation: 89,
      caller: "Don't tell me to calm down! Nobody here cares about me!",
    },
    {
      user: "I'm sorry, that was wrong of me. Losing your job like that must be really hard.",
      emotion: 'neutral',
      reasons: ['calm tone -8', 'de-escalating words -12'],
      delta: -20,
      agitation: 69,
      caller: "It is hard... I have a family. How do I tell my wife?",
    },
    {
      user: "I hear you. Your family matters, and you're not alone. Take your time, I'm here.",
      emotion: 'neutral',
      reasons: ['calm tone -8', 'de-escalating words -18'],
      delta: -20,
      agitation: 49,
      caller: "I didn't want any of this. I just got so angry.",
    },
    {
      user: "That makes sense. Nobody is going to hurt you. We can talk about your options together.",
      emotion: 'neutral',
      reasons: ['calm tone -8', 'de-escalating words -18'],
      delta: -20,
      agitation: 29,
      caller: "Okay... maybe there is another way.",
    },
    {
      user: "There is. Would you open the door so we can sit down and talk, just you and me?",
      emotion: 'happy',
      reasons: ['calm tone -8', 'de-escalating words -12'],
      delta: -20,
      agitation: 9,
      caller: "Alright. I'm opening the door. Thank you for listening.",
    },
  ],
};

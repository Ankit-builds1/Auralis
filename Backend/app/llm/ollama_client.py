"""
Streaming Llama 3 client for the live backend (Week 2: LLM Integration).

Based on Siddhant's ml/src/llm/ollama_client.py + emotion_prompt.py
(same persona, same short-reply rules), with three changes for the
real-time server:

  1. async (aiohttp) instead of requests, so it never blocks the
     WebRTC event loop and can be cancelled later (Week 4 interruptions)
  2. /api/chat with a short conversation history, so Auralis
     remembers the last few turns
  3. warmup() + keep_alive, so the model is already in memory when
     the first sentence arrives (otherwise TTFT includes model loading)
"""

import json
import os

import aiohttp


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Keep the model loaded in RAM between requests
KEEP_ALIVE = "30m"

# Short replies = faster replies (Siddhant's value)
MAX_TOKENS = 80
TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "You are Auralis, a concise and empathetic voice assistant. "
    "Respond naturally and empathetically. "
    "Keep every response to 1-2 short sentences. "
    "Make the first sentence short, no more than 8 words, "
    "so you can start speaking right away. "
    "Do not use bullet points, lists, emojis or markdown. "
    "Do not repeat the user's statement. "
    "Do not mention an emotion model or that you are an AI. "
    "If the user talks about hurting themselves, ending their life or "
    "not wanting to live, never treat it as anger or frustration: reply "
    "with warmth, ask gently if they are safe right now, and tell them "
    "they can call the Tele-MANAS helpline on 14416 at any time. "
    "You cannot call, message or contact anyone, so never offer to."
)

# Words that can mean the user is in crisis. The voice emotion model
# cannot hear this (it only hears tone), so we check the words too.
CRISIS_PHRASES = (
    "kill myself", "killing myself", "end my life", "ending my life",
    "suicide", "want to die", "wanna die", "hurt myself", "harm myself",
    "self harm", "self-harm", "bad step", "no reason to live",
    "better off dead", "don't want to live", "dont want to live",
)

CRISIS_HINT = (
    "Safety: the user may be thinking about harming themselves. "
    "Respond with care, ask if they are safe right now, and mention the "
    "Tele-MANAS helpline 14416. Do not call it anger."
)


def is_crisis(transcript):
    text = (transcript or "").lower()
    return any(phrase in text for phrase in CRISIS_PHRASES)


def build_messages(history, transcript, emotion=None, hint=None):
    """
    history:    list of {"role": "user"|"assistant", "content": str}
    transcript: what the user just said
    emotion:    optional label from the voice emotion model (e.g. "angry")
    hint:       optional reply-style instruction for that emotion group
    The emotion is treated as possibly wrong; the words come first.
    Crisis words always override the voice emotion hint.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    if is_crisis(transcript):
        user_content = f'User said: "{transcript}"\n{CRISIS_HINT}'
    elif emotion:
        user_content = (
            f'User said: "{transcript}"\n'
            f'Voice tone detected (may be imperfect): "{emotion}"'
        )
        if hint:
            user_content += f"\nReply style: {hint}"
    else:
        user_content = transcript

    messages.append({"role": "user", "content": user_content})
    return messages

async def stream_reply(messages, session=None):
    """
    Async generator: yields reply tokens as soon as Ollama produces them.

    Raises RuntimeError with a readable message if Ollama is not reachable.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120, sock_connect=10)
        )

    try:
        try:
            response = await session.post(f"{OLLAMA_URL}/api/chat", json=payload)
        except aiohttp.ClientConnectorError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {OLLAMA_URL}. Is 'ollama serve' running?"
            ) from exc

        async with response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"Ollama error {response.status}: {body[:200]}")

            # Ollama streams one JSON object per line
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                data = json.loads(line)

                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")

                token = data.get("message", {}).get("content", "")
                if token:
                    yield token

                if data.get("done"):
                    break
    finally:
        if own_session:
            await session.close()


async def warmup():
    """
    Load the model into memory once at server start.
    Without this, the first reply's TTFT includes several seconds of
    model loading.
    """
    messages = [{"role": "user", "content": "Say ok."}]
    try:
        async for _ in stream_reply(messages):
            pass
        print(f"[LLM] Warm-up done (model={OLLAMA_MODEL})")
    except Exception as exc:
        print(f"[LLM] Warm-up failed: {exc}")
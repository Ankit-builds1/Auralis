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
    "Do not use bullet points, lists, emojis or markdown. "
    "Do not repeat the user's statement. "
    "Do not mention an emotion model or that you are an AI."
)


def build_messages(history, transcript, emotion=None):
    """
    history:    list of {"role": "user"|"assistant", "content": str}
    transcript: what the user just said
    emotion:    optional label from the emotion model (e.g. "happy").
                Treated as possibly-wrong context, as in Siddhant's prompt.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)

    if emotion:
        user_content = (
            f'User said: "{transcript}"\n'
            f'Detected emotion (may be imperfect): "{emotion}"'
        )
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

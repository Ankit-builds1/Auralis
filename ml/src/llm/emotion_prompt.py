from src.llm.ollama_client import generate_response


def generate_emotion_aware_response(
    transcript: str,
    emotion: str,
):
    """
    Stream an emotion-aware response from Llama 3.
    Yields response tokens as they arrive.
    """

    prompt = f"""
You are Auralis, a concise and empathetic voice assistant.

User transcript:
"{transcript}"

Detected emotion:
"{emotion}"

Treat the detected emotion as potentially imperfect context.
Respond naturally and empathetically.
Keep the response to 1-2 short sentences.
Do not use bullet points.
Do not repeat the user's statement.
Do not mention the emotion model or that you are an AI.
"""

    yield from generate_response(prompt)


if __name__ == "__main__":
    for token in generate_emotion_aware_response(
        transcript="I finally finished my project and it went really well.",
        emotion="happy",
    ):
        print(token, end="", flush=True)

    print()


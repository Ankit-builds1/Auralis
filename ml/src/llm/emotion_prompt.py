from src.llm.ollama_client import generate_response


def generate_emotion_aware_response(
    transcript: str,
    emotion: str,
) -> str:
    """
    Generate a concise response using the user's transcript
    and detected emotion as contextual information.
    """

    prompt = f"""
You are Auralis, a concise and empathetic voice assistant.

The user's speech was transcribed as:
"{transcript}"

An emotion model detected:
"{emotion}"

The detected emotion may be imperfect, so treat it as contextual information,
not absolute truth.

Respond naturally to what the user said.
Use the detected emotion only when it helps make the response more appropriate.

Keep the response to 1-2 short sentences.
Do not use bullet points.
Do not repeat the user's statement.
Do not mention the emotion model.
Do not mention that you are an AI.
"""

    return generate_response(prompt)


if __name__ == "__main__":
    transcript = "I finally finished my project and it went really well."
    emotion = "happy"

    response = generate_emotion_aware_response(
        transcript,
        emotion,
    )

    print(response)

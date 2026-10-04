import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def generate_response(prompt: str) -> str:
    """
    Send a prompt to Llama 3 through Ollama
    and return the generated response.
    """

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 256,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["response"].strip()


if __name__ == "__main__":
    prompt = """
You are Auralis, a concise voice assistant.

The user says:
"I am feeling really stressed about my exams today."

Respond naturally and empathetically.
Keep the response to 1-2 short sentences.
Do not give long explanations.
Do not use bullet points.
Do not repeat the user's statement.
"""

    result = generate_response(prompt)
    print(result)

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def generate_response(prompt: str):
    """
    Stream response tokens from Llama 3 through Ollama.
    Yields each token as soon as it arrives.
    """

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "num_predict": 80,
        },
    }

    with requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=(10, 120),
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            data = json.loads(line)
            token = data.get("response", "")

            if token:
                yield token

            if data.get("done", False):
                break


if __name__ == "__main__":
    prompt = (
        "You are Auralis, a concise voice assistant. "
        "Respond empathetically to someone stressed about exams. "
        "Use 1-2 short sentences."
    )

    for token in generate_response(prompt):
        print(token, end="", flush=True)

    print()


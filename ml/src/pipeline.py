from src.models.audio_emotion import predict_emotion_from_audio
from src.llm.emotion_prompt import generate_emotion_aware_response


def process_input(audio_path: str, transcript: str):
    """
    Complete Auralis ML pipeline.

    Audio:
        audio -> emotion

    Text:
        transcript + emotion -> Llama response
    """

    emotion = predict_emotion_from_audio(audio_path)

    response = generate_emotion_aware_response(
        transcript=transcript,
        emotion=emotion,
    )

    return {
        "emotion": emotion,
        "response": response,
    }


if __name__ == "__main__":
    audio_path = "data/raw/CREMA-D/AudioWAV/1001_DFA_HAP_XX.wav"

    transcript = (
        "I finally finished my project and it went really well."
    )

    result = process_input(
        audio_path=audio_path,
        transcript=transcript,
    )

    print("\nAuralis ML Pipeline")
    print("=" * 40)
    print("Emotion:", result["emotion"])
    print("Response:", result["response"])

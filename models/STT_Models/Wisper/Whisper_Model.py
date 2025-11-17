import whisper
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

import whisper


def transcribe_audio(audio_path: str, model_name: str = "small") -> str:
    """
    Transcribes an audio file using OpenAI Whisper.

    Parameters:
        audio_path (str): Path to the audio file (e.g., '1.mp3').
        model_name (str): Whisper model name ('tiny', 'base', 'small', 'medium', 'large').

    Returns:
        str: Transcribed text from the audio.
    """
    # Load Whisper model
    model = whisper.load_model(model_name)

    # Run transcription (force FP32 if on CPU)
    result = model.transcribe(audio_path, fp16=False)

    # Return the clean text
    return result["text"].strip()



text = transcribe_audio("1.mp3", model_name="small")
print("Transcribed Text:\n", text)
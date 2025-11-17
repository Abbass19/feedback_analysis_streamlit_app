from models.Classification_Model.BERT_MultiHead_Classifier.BERT_Classification_Model import Classification_Sentiment_Array
from models.NER_Model.GLiNER.GLiNER_NER_Model import extract_entities_dict
import whisper

import warnings
import transformers
import torch

# Suppress FutureWarnings from Transformers / PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Optional: reduce Transformers logging
transformers.logging.set_verbosity_error()
torch.set_printoptions(profile="full")  # optional if you also want cleaner tensor printing

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def text_to_record(text):
    """
    Convert a single text to a merged record:
    [text, 5 sentiment numbers, 12 NER strings]
    Missing NER values are replaced with empty string.
    """
    # 1️⃣ Feedback predictions (list of 5 numbers)
    feedback_preds = Classification_Sentiment_Array(text)

    # 2️⃣ NER predictions (list of 12 strings)
    ner_preds = extract_entities_dict(text)

    # Replace None with empty string
    ner_preds_clean = [val if val is not None else " " for val in ner_preds]

    # 3️⃣ Merge everything into a single record
    record = [text] + feedback_preds + ner_preds_clean
    return record

def ai_classifier(text):
    return Classification_Sentiment_Array(text)

def ai_ner(text):
    return extract_entities_dict(text)

def ai_stt(audio_path: str, model_name: str = "small") -> str:
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



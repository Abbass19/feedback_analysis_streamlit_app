import os
import re
import pandas as pd
from difflib import SequenceMatcher
import whisper  # make sure Whisper is installed


# ---------------- Text Processing ----------------
def clean_text(text):
    text = re.sub(r'[^\w\s\u0621-\u064A]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def word_tokenize(text):
    return text.split()


def compare_texts(original_words, transcribed_words):
    matches = sum(1 for w1, w2 in zip(original_words, transcribed_words) if w1 == w2)
    accuracy = matches / max(len(original_words), 1)

    ratio = SequenceMatcher(None, ' '.join(original_words), ' '.join(transcribed_words)).ratio()

    original_set = set(original_words)
    transcribed_set = set(transcribed_words)
    tp = len(original_set & transcribed_set)
    precision = tp / len(transcribed_set) if transcribed_set else 0
    recall = tp / len(original_set) if original_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return accuracy, ratio, precision, recall, f1


# ---------------- Transcription ----------------
# Load model once
model = whisper.load_model("small")


def transcribe_audio_file(audio_path):
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


# ---------------- Evaluation ----------------
def evaluate_whisper_on_HEAR():
    # Absolute path to your dataset folder
    folder_path = r"/models\STT_Models\Wisper\Audio_Dataset_First_20"
    csv_file = os.path.join(folder_path, "HEAR_Dataset.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at {csv_file}")

    df = pd.read_csv(csv_file)
    df = df.iloc[1:21]  # skip header row, take first 20

    results = []
    for i, text in enumerate(df.iloc[:, 0], start=1):
        clean_original = clean_text(str(text))
        original_words = word_tokenize(clean_original)

        audio_file = os.path.join(folder_path, f"{i}.mp3")
        if not os.path.exists(audio_file):
            print(f"⚠️ Missing file: {audio_file}")
            continue

        print(f"Transcribing file {i}.mp3 ...")
        transcribed_text = transcribe_audio_file(audio_file)
        print(f"Done: {transcribed_text}")

        clean_transcribed = clean_text(transcribed_text)
        transcribed_words = word_tokenize(clean_transcribed)

        accuracy, ratio, precision, recall, f1 = compare_texts(original_words, transcribed_words)

        results.append({
            "Audio": f"{i}.mp3",
            "Original": clean_original,
            "Transcribed": clean_transcribed,
            "Accuracy": round(accuracy, 3),
            "Levenshtein": round(ratio, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel(os.path.join(folder_path, "whisper_evaluation_results.xlsx"), index=False)
    print("\n✅ Results saved to whisper_evaluation_results.xlsx")
    print(results_df)


if __name__ == "__main__":
    evaluate_whisper_on_HEAR()

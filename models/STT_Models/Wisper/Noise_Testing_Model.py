import os
import random
import pandas as pd
import torch
import whisper
from difflib import SequenceMatcher
from pydub import AudioSegment
from pydub.generators import WhiteNoise
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
folder_path = r"/models\STT_Models\Wisper\Audio_Dataset_First_20"
csv_file = os.path.join(folder_path, "HEAR_Dataset.csv")

# ---------------- Load Texts ----------------
df = pd.read_csv(csv_file)
texts = df.iloc[1:21, 0].tolist()  # first 20 entries

# ---------------- Load Whisper ----------------
model = whisper.load_model("small")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Helper Functions ----------------
def add_noise_to_audio(audio_path, noise_level):
    audio = AudioSegment.from_file(audio_path)
    noise = WhiteNoise().to_audio_segment(duration=len(audio))
    noise = noise - (30 - noise_level*30)  # adjust volume
    return audio.overlay(noise)

def transcribe_audio(audio_segment):
    # save temp file
    temp_path = "temp.wav"
    audio_segment.export(temp_path, format="wav")
    result = model.transcribe(temp_path, fp16=False)
    return result["text"]

def clean_text(text):
    return ''.join(c for c in text if c.isalnum() or c.isspace())

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

# ---------------- Noise Levels ----------------
noise_levels = [i/100 for i in range(5, 101, 5)]
metrics = {"Accuracy": [], "Levenshtein": [], "Precision": [], "Recall": [], "F1": []}

# ---------------- Run Noise Test ----------------
for nl in noise_levels:
    acc_list, lev_list, prec_list, rec_list, f1_list = [], [], [], [], []
    for i, text in enumerate(texts, start=1):
        audio_file = os.path.join(folder_path, f"{i}.mp3")
        if not os.path.exists(audio_file):
            print(f"⚠️ Missing file: {audio_file}")
            continue

        noisy_audio = add_noise_to_audio(audio_file, nl)
        transcribed_text = transcribe_audio(noisy_audio)

        acc, lev, prec, rec, f1 = compare_texts(word_tokenize(clean_text(text)),
                                                word_tokenize(clean_text(transcribed_text)))

        acc_list.append(acc)
        lev_list.append(lev)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    metrics["Accuracy"].append(sum(acc_list)/len(acc_list))
    metrics["Levenshtein"].append(sum(lev_list)/len(lev_list))
    metrics["Precision"].append(sum(prec_list)/len(prec_list))
    metrics["Recall"].append(sum(rec_list)/len(rec_list))
    metrics["F1"].append(sum(f1_list)/len(f1_list))

# ---------------- Plot ----------------
plt.figure(figsize=(10,6))
for key in ["Accuracy", "Levenshtein", "Precision", "Recall", "F1"]:
    plt.plot(range(5, 101, 5), metrics[key], marker='o', label=key)
plt.title("STT Metrics vs Noise Level (%)")
plt.xlabel("Noise Level (%)")
plt.ylabel("Metric Value")
plt.ylim(0,1.05)
plt.grid(True)
plt.legend()
plt.show()

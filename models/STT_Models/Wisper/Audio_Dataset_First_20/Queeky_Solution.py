import os
import re
import pandas as pd
from difflib import SequenceMatcher

# ---------------- Text processing ----------------
def clean_text(text):
    text = re.sub(r'[^\w\s\u0621-\u064A]', '', str(text))
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

# ---------------- Evaluation ----------------
def evaluate_from_excel(excel_file):
    folder_path = os.path.dirname(os.path.abspath(excel_file))
    df = pd.read_excel(excel_file)

    results = []
    for i, row in df.iterrows():
        original_text = clean_text(row['Original'])
        transcribed_text = clean_text(row['Transcribed'])

        original_words = word_tokenize(original_text)
        transcribed_words = word_tokenize(transcribed_text)

        accuracy, ratio, precision, recall, f1 = compare_texts(original_words, transcribed_words)

        results.append({
            "Audio": row.get('Audio', f"Record_{i+1}"),
            "Accuracy": round(accuracy, 3),
            "Levenshtein": round(ratio, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3)
        })

    results_df = pd.DataFrame(results)
    # Compute overall averages
    avg_metrics = results_df[["Accuracy", "Levenshtein", "Precision", "Recall", "F1"]].mean()

    # Save report as TXT
    report_file = os.path.join(folder_path, "../Testing_Procedure_Report.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("Whisper Evaluation Report\n")
        f.write("========================\n\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\nOverall Averages:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.3f}\n")

    print(f"✅ Evaluation report saved to: {report_file}")

if __name__ == "__main__":
    excel_file = "whisper_evaluation_results.xlsx"  # place next to this script
    evaluate_from_excel(excel_file)

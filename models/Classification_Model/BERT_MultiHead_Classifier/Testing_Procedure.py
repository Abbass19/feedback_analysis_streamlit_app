# test_feedback_model.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from BERT_Classification_Model import predict_feedback, ASPECT_NAMES

# ----------------------------
# 1️⃣ Load the testing dataset
# ----------------------------
df = pd.read_csv("testing_dataset.csv")  # make sure path is correct
texts = df["Review Text"].tolist()
labels_array = df[["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"]].values

# ----------------------------
# 2️⃣ Run predictions
# ----------------------------
all_preds = []
all_probs = []  # if you want probabilities, we can extend predict_feedback later

for text in texts:
    preds = predict_feedback([text])
    all_preds.append([preds[aspect] for aspect in ASPECT_NAMES])

all_preds = np.array(all_preds)
all_labels = labels_array

# ----------------------------
# 3️⃣ Compute metrics for each aspect
# ----------------------------
metrics = []
for i, aspect in enumerate(ASPECT_NAMES):
    y_true = all_labels[:, i]
    y_pred = all_preds[:, i]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # For mAP we need probabilities per class; for now, we can compute a crude version using one-hot predictions
    y_true_onehot = np.zeros((len(y_true), 4))
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    y_prob_onehot = np.zeros_like(y_true_onehot)
    y_prob_onehot[np.arange(len(y_pred)), y_pred] = 1
    try:
        mAP = average_precision_score(y_true_onehot, y_prob_onehot, average='macro')
    except ValueError:
        mAP = 0.0

    metrics.append({
        "Aspect": aspect,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "mAP": mAP
    })

# ----------------------------
# 4️⃣ Print metrics
# ----------------------------
for m in metrics:
    print(f"{m['Aspect']}")
    print(f"  Accuracy: {m['Accuracy']:.4f}")
    print(f"  Precision: {m['Precision']:.4f}")
    print(f"  Recall: {m['Recall']:.4f}")
    print(f"  F1-score: {m['F1']:.4f}")
    print(f"  mAP: {m['mAP']:.4f}")
    print("-" * 50)

mean_f1 = np.mean([m['F1'] for m in metrics])
mean_map = np.mean([m['mAP'] for m in metrics])
print(f"\nOverall Mean F1: {mean_f1:.4f}")
print(f"Overall Mean mAP: {mean_map:.4f}")

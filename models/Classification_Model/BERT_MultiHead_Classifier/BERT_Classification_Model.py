import os
import torch
from torch import nn
from transformers import BertModel, AutoTokenizer

import os
import requests
import zipfile

MODEL_URL = "https://huggingface.co/abbasszahreddine/feedback-classifier-model/blob/main/feedback_classifier.pt"
TOKENIZER_URL = "https://huggingface.co/abbasszahreddine/feedback-classifier-model/blob/main/saved_model.zip"

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {dest_path}")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        download_file(MODEL_URL, MODEL_PATH)
    if not os.path.exists(TOKENIZER_PATH):
        zip_path = os.path.join(os.path.dirname(TOKENIZER_PATH), "tokenizer.zip")
        download_file(TOKENIZER_URL, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TOKENIZER_PATH)
        os.remove(zip_path)



# ----------------------------
# Model definition
# ----------------------------
class MultiHeadClassifier(nn.Module):
    def __init__(self, bert_model_name="aubmindlab/bert-base-arabertv2", num_heads=5, num_classes=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            bert_model_name,
            torch_dtype=torch.float32
        )
        hidden_size = self.bert.config.hidden_size
        self.heads = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_heads)])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = [head(cls_embedding) for head in self.heads]
        return logits

# ----------------------------
# Global paths and variables
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "saved_model/feedback_classifier.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "saved_model/tokenizer")

model = None
tokenizer = None

ASPECT_NAMES = ["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"]

# ----------------------------
# Function: predict feedback as array with lazy-loading
# ----------------------------
def Classification_Sentiment_Array(text_list):
    """
    Predicts feedback for a list of texts.
    Returns a list of predicted class indices in the same order as ASPECT_NAMES.
    Lazy-loads the model and tokenizer on first call.
    """
    global model, tokenizer
    ensure_model()

    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        model = MultiHeadClassifier()
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"), strict=False)
        model.eval()

    if isinstance(text_list, str):
        text_list = [text_list]

    encoding = tokenizer(text_list, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])

    # Convert logits to predicted class indices
    predictions = [torch.argmax(l, dim=1).tolist() for l in logits]  # list of lists

    # Transpose so each row = text, columns = aspects
    predictions_array = [list(i) for i in zip(*predictions)]
    return predictions_array if len(predictions_array) > 1 else predictions_array[0]

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    example_text = "المستشفى مستشفى رائع، الخدمة عظيمة جدا وممتازة ولكن السعر غالي جدا. الخدمة العاجلة سريعة جدا وممتازة. الطاقم الطبي ممتاز"
    result = Classification_Sentiment_Array(example_text)
    print(result)



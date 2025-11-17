import os
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import requests
import json



# New
TOKENIZER_URL = "https://huggingface.co/abbasszahreddine/feedback_analysis_streamlit_app/raw/main/tokenizer.json"
TOKENIZER_CONFIG_URL = "https://huggingface.co/abbasszahreddine/feedback_analysis_streamlit_app/raw/main/tokenizer_config.json"
SPECIAL_TOKENS_URL = "https://huggingface.co/abbasszahreddine/feedback_analysis_streamlit_app/raw/main/special_tokens_map.json"
VOCAB_URL = "https://huggingface.co/abbasszahreddine/feedback_analysis_streamlit_app/raw/main/vocab.txt"
CONFIG_URL = "https://huggingface.co/abbasszahreddine/feedback_analysis_streamlit_app/raw/main/model_config.json"


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
    """Download and setup model files if they don't exist"""
    os.makedirs(TOKENIZER_PATH, exist_ok=True)

    tokenizer_file = os.path.join(TOKENIZER_PATH, "tokenizer.json")
    if not os.path.exists(tokenizer_file):
        download_file(TOKENIZER_URL, tokenizer_file)

    tokenizer_config_file = os.path.join(TOKENIZER_PATH, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_file):
        download_file(TOKENIZER_CONFIG_URL, tokenizer_config_file)

    special_tokens_file = os.path.join(TOKENIZER_PATH, "special_tokens_map.json")
    if not os.path.exists(special_tokens_file):
        download_file(SPECIAL_TOKENS_URL, special_tokens_file)

    vocab_file = os.path.join(TOKENIZER_PATH, "vocab.txt")
    if not os.path.exists(vocab_file):
        download_file(VOCAB_URL, vocab_file)

    # Download config file if it exists
    config_path = os.path.join(os.path.dirname(MODEL_PATH), "model_config.json")
    if not os.path.exists(config_path):
        try:
            download_file(CONFIG_URL, config_path)
        except:
            print("Config file not available, using default parameters")


# ----------------------------
# UPDATED Model definition for enhanced architecture
# ----------------------------
class MultiHeadClassifier(nn.Module):
    def __init__(self, bert_model_name="aubmindlab/bert-base-arabertv2", num_heads=5, num_classes=4, dropout_rate=0.3):
        super().__init__()
        # Compatible loading for Python 3.13.9
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")


        hidden_size = self.bert.config.hidden_size

        # Enhanced architecture with dropout and hidden layers
        self.dropout = nn.Dropout(dropout_rate)

        # Updated head architecture matching the training script
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
            ) for _ in range(num_heads)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)  # Apply dropout
        logits = [head(cls_embedding) for head in self.heads]
        return logits


# ----------------------------
# Global paths and variables
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "feedback_classifier.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "saved_model", "tokenizer")
CONFIG_PATH = os.path.join(BASE_DIR, "saved_model", "model_config.json")

model = None
tokenizer = None

ASPECT_NAMES = ["Pricing", "Appointments", "Medical Staff", "Customer Service", "Emergency Services"]


# ----------------------------
# Enhanced prediction function with confidence scores
# ----------------------------
def Classification_Sentiment_Array(text_list, return_confidence=False):
    """
    Predicts feedback for a list of texts.

    Args:
        text_list (str or list): Input text or list of texts
        return_confidence (bool): If True, returns confidence scores along with predictions

    Returns:
        If return_confidence=False: list of predicted class indices in same order as ASPECT_NAMES
        If return_confidence=True: tuple of (predictions, confidence_scores)
    """
    global model, tokenizer
    ensure_model()

    if model is None or tokenizer is None:
        print("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        # Load model configuration if available
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            model = MultiHeadClassifier(
                bert_model_name=config.get("bert_model_name", "aubmindlab/bert-base-arabertv2"),
                num_heads=config.get("num_heads", 5),
                num_classes=config.get("num_classes", 4),
                dropout_rate=config.get("dropout_rate", 0.3)
            )
        else:
            # Fallback to default parameters
            model = MultiHeadClassifier()

        # Load model weights with enhanced error handling
        try:
            state_dict = torch.load(MODEL_PATH, map_location="cpu")

            # Handle potential DataParallel wrapping
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v  # remove 'module.' prefix
                else:
                    new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            model.eval()
            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Try direct loading as fallback
            try:
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                print("✅ Model loaded with strict=False")
            except Exception as e2:
                print(f"❌ Fallback loading also failed: {e2}")
                raise

    if isinstance(text_list, str):
        text_list = [text_list]

    # Tokenize with proper settings
    encoding = tokenizer(
        text_list,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])

    # Convert logits to predicted class indices and confidence scores
    predictions = []
    confidence_scores = []

    for head_logits in logits:
        probs = torch.softmax(head_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        confs = torch.max(probs, dim=1).values

        predictions.append(preds.tolist())
        confidence_scores.append(confs.tolist())

    # Transpose so each row = text, columns = aspects
    predictions_array = [list(i) for i in zip(*predictions)]
    confidence_array = [list(i) for i in zip(*confidence_scores)] if return_confidence else None

    if return_confidence:
        if len(predictions_array) == 1:
            return predictions_array[0], confidence_array[0]
        return predictions_array, confidence_array
    else:
        return predictions_array if len(predictions_array) > 1 else predictions_array[0]


def Classification_Sentiment_Detailed(text_list):
    """
    Returns detailed predictions including aspect names and confidence scores.

    Returns:
        List of dictionaries with detailed predictions for each text
    """
    predictions, confidences = Classification_Sentiment_Array(text_list, return_confidence=True)

    if isinstance(text_list, str):
        predictions = [predictions]
        confidences = [confidences]
        text_list = [text_list]

    detailed_results = []
    for i, text in enumerate(text_list):
        result = {"text": text, "aspects": {}}
        for j, aspect in enumerate(ASPECT_NAMES):
            pred_idx = predictions[i][j] if i < len(predictions) else predictions[j]
            conf = confidences[i][j] if i < len(confidences) else confidences[j]

            # Map prediction index to label
            label_map = {0: "negative", 1: "neutral", 2: "positive", 3: "unmentioned"}
            label = label_map.get(pred_idx, "unknown")

            result["aspects"][aspect] = {
                "sentiment": label,
                "confidence": round(conf, 4),
                "class_index": pred_idx
            }
        detailed_results.append(result)

    return detailed_results[0] if len(detailed_results) == 1 else detailed_results


# ----------------------------
# Example usage and testing
# ----------------------------
if __name__ == "__main__":
    # Test examples
    example_texts = [
        "المستشفى مستشفى رائع، الخدمة عظيمة جدا وممتازة ولكن السعر غالي جدا. الخدمة العاجلة سريعة جدا وممتازة. الطاقم الطبي ممتاز",
        "المواعيد صعبة والحجز يستغرق وقت طويل، لكن الأطباء متعاونون جدا",
        "أسعار معقولة وخدمة العملاء جيدة"
    ]

    print("🧪 Testing Basic Prediction:")
    result = Classification_Sentiment_Array(example_texts[0])
    print(f"Input: {example_texts[0]}")
    print(f"Predictions: {result}")
    print()

    print("🧪 Testing Multiple Texts:")
    results = Classification_Sentiment_Array(example_texts)
    for i, (text, pred) in enumerate(zip(example_texts, results)):
        print(f"Text {i + 1}: {pred}")
    print()

    print("🧪 Testing Detailed Output:")
    detailed = Classification_Sentiment_Detailed(example_texts[0])
    print("Detailed Results:")
    for aspect, info in detailed["aspects"].items():
        print(f"  {aspect}: {info['sentiment']} (confidence: {info['confidence']})")
    print()

    print("🧪 Testing with Confidence Scores:")
    preds, confs = Classification_Sentiment_Array(example_texts, return_confidence=True)
    for i, (text, pred, conf) in enumerate(zip(example_texts, preds, confs)):
        print(f"Text {i + 1}:")
        for j, aspect in enumerate(ASPECT_NAMES):
            print(f"  {aspect}: {pred[j]} (confidence: {conf[j]:.3f})")
import warnings
import transformers
from gliner import GLiNER
import os

# ----------------------------
# Suppress warnings
# ----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
transformers.logging.set_verbosity_error()

# ----------------------------
# Global variables (lazy-loaded)
# ----------------------------
model = None
DEFAULT_ENTITY_LABELS = [
    "اسم_الطبيب", "الدور_الوظيفي", "اسم_المستشفى", "القسم",
    "التخصص", "منطقة_الخدمة", "السعر", "الوقت",
    "الموقع", "جودة_الخدمة", "نوع_المشكلة", "نوع_العلاج"
]

# Mapping Arabic labels → English keys
LABEL_MAP = {
    "اسم_الطبيب": "doctor_name",
    "الدور_الوظيفي": "staff_role",
    "اسم_المستشفى": "hospital_name",
    "القسم": "department",
    "التخصص": "specialty",
    "منطقة_الخدمة": "service_area",
    "السعر": "price",
    "الوقت": "time_expression",
    "الموقع": "location",
    "جودة_الخدمة": "quality_aspect",
    "نوع_المشكلة": "issue_type",
    "نوع_العلاج": "treatment_type"
}

# ----------------------------
# Function: extract entities as dictionary
# ----------------------------
def extract_entities_dict(text, labels=None, threshold=0.65):
    """
    Extract named entities from text using GLiNER.
    Returns a dictionary with English keys and extracted entity values.
    Missing entities are returned as empty strings.
    """
    global model

    # Lazy-load model
    if model is None:
        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

    if labels is None:
        labels = DEFAULT_ENTITY_LABELS

    # Predict entities using Arabic labels
    entities_raw = model.predict_entities(text, labels=labels, threshold=threshold)
    entities_temp = {}

    for e in entities_raw:
        # Keep first occurrence only
        if e["label"] not in entities_temp:
            entities_temp[e["label"]] = e["text"]

    # Convert to English-keyed dictionary
    entities_dict = {LABEL_MAP[label]: entities_temp.get(label, "") for label in labels}

    return entities_dict

text = """
زار السيد أحمد علي مستشفى الرحمة يوم الثلاثاء 12 نوفمبر 2025 لتقييم الخدمة. 
استقبله الدكتور محمد جمال في قسم الطب الباطني، وكانت جودة الخدمة ممتازة. 
تكلفة العلاج كانت 5000 ليرة، والوقت المستغرق في الفحص حوالي 30 دقيقة. 
العيادة تقع في منطقة الأشرفية، وكان التخصص دقيقاً جداً في التشخيص.
"""


print(extract_entities_dict(text))
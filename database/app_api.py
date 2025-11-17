from typing_extensions import Counter

from database.db_layer import db_add_record, db_get_with_sentiment,db_clear_records
from models.ai_api import ai_ner,ai_stt,ai_classifier
import pandas as pd
from typing import List, Dict, Any


def api_add(text: str):
    """
    Processes a single text with sentiment and NER models
    and inserts it into the PostgreSQL database.
    """
    # 1️⃣ Predict sentiment scores (list of 5 numbers)
    sentiment_preds = ai_classifier(text)

    # Map to the database columns
    sentiment_scores = {
        "pricing": sentiment_preds[0],
        "appointments": sentiment_preds[1],
        "staff": sentiment_preds[2],
        "customer_service": sentiment_preds[3],
        "emergency_services": sentiment_preds[4]
    }

    # 2️⃣ Predict NER values (list of 12 strings)
    ner_preds = ai_ner(text)

    # Replace None with empty string and map to database columns
    ner_values = {
        "doctor_name": ner_preds.get("doctor_name", ""),
        "staff_role": ner_preds.get("staff_role", ""),
        "hospital_name": ner_preds.get("hospital_name", ""),
        "department": ner_preds.get("department", ""),
        "specialty": ner_preds.get("specialty", ""),
        "service_area": ner_preds.get("service_area", ""),
        "price": ner_preds.get("price", ""),
        "time_expression": ner_preds.get("time_expression", ""),
        "location": ner_preds.get("location", ""),
        "quality_aspect": ner_preds.get("quality_aspect", ""),
        "issue_type": ner_preds.get("issue_type", ""),
        "treatment_type": ner_preds.get("treatment_type", "")
    }

    # 3️⃣ Insert into database
    db_add_record(text, sentiment_scores, ner_values)

def api_query(selected_sentiments, selected_classes):
    """
    Intermediate function between the Streamlit UI and the database layer.

    - Maps sentiment & classification names to DB columns/values.
    - Applies Union logic for selected sentiments (Positive OR Negative).
    - Applies Intersection logic for classifications (Pricing AND Appointments).
    - Retrieves results ordered by the latest 'created_at' timestamp first.
    """

    # Local mapping for sentiment labels to database integer values
    SENTIMENT_MAP = {
        "Negative": 0,
        "Positive": 1,
        "Neutral": 2
    }

    # Local mapping for classification to column names
    CLASSIFICATION_MAP = {
        "Pricing": "sentiment_pricing",
        "Appointments": "sentiment_appointments",
        "Medical Staff": "sentiment_staff",
        "Customer Service": "sentiment_customer_service",
        "Emergency Services": "sentiment_emergency_services"
    }

    # Convert UI sentiment selections to database values
    sentiment_values = [SENTIMENT_MAP[s] for s in selected_sentiments if s in SENTIMENT_MAP]

    # Default: if nothing selected, include all except "not mentioned"
    if not sentiment_values:
        sentiment_values = [0, 1, 2]

    # Build the sentiment filter dictionary for the DB API function
    sentiment_filter = {}
    for cls in selected_classes:
        col_name = CLASSIFICATION_MAP.get(cls)
        if col_name:
            sentiment_filter[col_name] = sentiment_values

    # If user selected no classification, apply to all sentiment columns
    if not sentiment_filter:
        for col_name in CLASSIFICATION_MAP.values():
            sentiment_filter[col_name] = sentiment_values

    # Call the database API
    records = db_get_with_sentiment(sentiment_filter)

    # Sort newest first (just to be absolutely sure)
    records = sorted(records, key=lambda x: x["created_at"], reverse=True)

    return records

def api_df_from_records(records: List[Dict[str, Any]]) -> pd.DataFrame:

    if not records:
        return pd.DataFrame()  # empty DataFrame if no records

    # Flatten records for NER dicts if needed
    processed_records = []
    for r in records:
        rec = r.copy()
        # Optionally flatten NER dict as string or leave as dict
        rec["ner"] = r.get("ner", {})
        processed_records.append(rec)

    df = pd.DataFrame(processed_records)
    return df

def api_clear_feedback() -> None:
    db_clear_records()

def api_sentiment_distribution(class_column: str = "sentiment_pricing") -> Dict[int, int]:
    """
    Returns a dictionary of sentiment value counts for a given sentiment column.
    Used for pie chart visualization.

    :param class_column: one of the 5 sentiment columns
    :return: dict where key=sentiment value (0,1,2) and value=count
    """
    # Prepare a filter with all possible values to fetch all records
    sentiment_filter = {class_column: [0, 1, 2]}
    records = db_get_with_sentiment(sentiment_filter)

    counts = {}
    for r in records:
        val = r.get(class_column)
        if val is not None:
            counts[val] = counts.get(val, 0) + 1

    return counts

def api_classification_distribution(class_column: str) -> Dict[str, int]:
    """
    Returns counts of sentiment values (0,1,2) for a given classification column.
    Example columns: sentiment_pricing, sentiment_staff, etc.
    """
    # Fetch all records for the given column
    records = db_get_with_sentiment({class_column: [0, 1, 2]})

    if not records:
        return {}

    # Count the occurrences of each sentiment value
    values = [r.get(class_column) for r in records if r.get(class_column) is not None]
    counts = dict(Counter(values))

    # Ensure all possible keys 0,1,2 exist
    for k in [0, 1, 2]:
        if k not in counts:
            counts[k] = 0

    return counts
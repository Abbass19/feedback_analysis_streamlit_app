from db_layer import db_add_record
from datetime import datetime

# Example feedback
feedback_text = "المستشفى ممتاز، الخدمة جيدة."
sentiment_scores = {
    "pricing": 2,
    "appointments": 1,
    "staff": 2,
    "customer_service": 2,
    "emergency_services": 0
}
ner_values = {
    "doctor_name": "د. أحمد علي",
    "staff_role": "ممرضة",
    "hospital_name": "مستشفى السلام",
    "department": "عيادات خارجية",
    "specialty": None,
    "service_area": None,
    "price": None,
    "time_expression": None,
    "location": None,
    "quality_aspect": None,
    "issue_type": None,
    "treatment_type": None
}

# Call the insert function
db_add_record(feedback_text, sentiment_scores, ner_values)
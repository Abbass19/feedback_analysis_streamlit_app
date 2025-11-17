from models.ai_api import ai_classifier,ai_ner,ai_stt

import warnings
import transformers
import torch

# Suppress FutureWarnings from Transformers / PyTorch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Optional: reduce Transformers logging
transformers.logging.set_verbosity_error()
torch.set_printoptions(profile="full")  # optional if you also want cleaner tensor printing

Feedback = "صراحه اعجز وصفي عن الدكتور حسام محمد هاجرس دكتور ممتاز ويده خفيفه وشغله صدق صدق جبار تقييمه فوق التقيييم الله يوفقه ( اسنان واعصاب )"

# print(ai_classifier(Feedback))
# print("Classifier Model Worked !!!")
# print(ai_ner(Feedback))
# print("NER Model Worked !!!")
# print(ai_stt("1.mp3"))
# print(f"STT Model Worked :) ")


# feedback_text = "اشكر ادارة مستشفى الدكتور سليمان الحبيب فرع الريان وشكر اي للاستاذ نايف السويحل لخدمتي وحل مشكلتي"
# print(ai_ner(feedback_text))
#
# another_one = "صراحه اعجز وصفي عن الدكتور حسام محمد هاجرس دكتور ممتاز ويده خفيفه وشغله صدق صدق جبار تقييمه فوق التقيييم الله يوفقه ( اسنان واعصاب )"



print(ai_classifier(Feedback))
print("Classifier Model Worked !!!")
print(ai_ner(Feedback))
print("NER Model Worked !!!")
print(ai_stt("1.mp3"))
print(f"STT Model Worked :) ")


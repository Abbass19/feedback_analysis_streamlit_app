from gliner import GLiNER
import os

os.environ["HUGGINGFACE_HUB_TOKEN"] = "YOUR_API_KEY"

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

text = """
زار السيد أحمد علي مستشفى الرحمة يوم الثلاثاء 12 نوفمبر 2025 لتقييم الخدمة. 
استقبله الدكتور محمد جمال في قسم الطب الباطني، وكانت جودة الخدمة ممتازة. 
تكلفة العلاج كانت 5000 ليرة، والوقت المستغرق في الفحص حوالي 30 دقيقة. 
العيادة تقع في منطقة الأشرفية، وكان التخصص دقيقاً جداً في التشخيص.
"""


DEFAULT_ENTITY_LABELS = [
    "اسم_الطبيب", "الدور_الوظيفي", "اسم_المستشفى", "القسم",
    "التخصص", "منطقة_الخدمة", "السعر", "الوقت",
    "الموقع", "جودة_الخدمة", "نوع_المشكلة", "نوع_العلاج"
]

entities = model.predict_entities(text, labels=DEFAULT_ENTITY_LABELS, threshold=0.65)

print(entities)

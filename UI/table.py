class FeedbackTableFormatter:
    SENTIMENT_MAP = {
        0: "سلبي",
        1: "إيجابي",
        2: "محايد"
    }

    def __init__(self, df):
        self.df = df.copy()

    def sentiment_color(self, val):
        if val == "إيجابي":
            return "background-color: #d4edda; color: #155724;"
        elif val == "سلبي":
            return "background-color: #f8d7da; color: #721c24;"
        else:
            return "background-color: #fff3cd; color: #856404;"

    def format_table(self):
        # Reorder and rename columns
        cols_order = [
            "feedback_text",
            "sentiment_pricing",
            "sentiment_appointments",
            "sentiment_customer_service",
            "sentiment_emergency_services"
        ]

        self.df = self.df[cols_order].copy()

        self.df.rename(columns={
            "feedback_text": "النص",  # match the actual column name
            "sentiment_pricing": "التسعير",
            "sentiment_appointments": "المواعيد",
            "sentiment_staff": "طاقم العمل",
            "sentiment_customer_service": "خدمة العملاء",
            "sentiment_emergency_services": "خدمات الطوارئ"
        }, inplace=True)

        # Map numeric sentiment to words
        for col in self.df.columns[1:]:
            self.df[col] = self.df[col].map(self.SENTIMENT_MAP).fillna("محايد")

        # Apply styling
        styled_df = self.df.style.applymap(self.sentiment_color, subset=self.df.columns[1:]) \
                                 .set_properties(**{"text-align": "right"}) \
                                 .set_table_styles([{"selector": "th", "props": [("text-align", "right")]}])
        return styled_df

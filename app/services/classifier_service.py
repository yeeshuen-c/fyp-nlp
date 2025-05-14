import joblib
from ml.preprocessing import clean_text, transform_text
from ml.cvtrain import detect_scam_type_by_keywords  # If using rule-based matcher

# Load vectorizer and model only once
vectorizer = joblib.load("ml/scamtype/vectorizer.pkl")
model = joblib.load("ml/scamtype/logistic_regression_model.pkl")

def classify_text(text: str) -> str:
    cleaned_text = clean_text(text)

    # First: Rule-based keyword match
    rule_label = detect_scam_type_by_keywords(cleaned_text)
    if rule_label:
        return rule_label

    # Else: ML-based prediction
    transformed = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed)[0]
    return prediction

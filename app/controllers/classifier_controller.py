from app.services.classifier_service import classify_text

def handle_classification_request(user_text: str) -> dict:
    result = classify_text(user_text)
    return {"scam_type": result}

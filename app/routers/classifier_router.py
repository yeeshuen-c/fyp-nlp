from fastapi import APIRouter
from pydantic import BaseModel
from ..controllers.classifier_controller import handle_classification_request

router = APIRouter()

class TextInput(BaseModel):
    content: str

@router.post("/classify")
def classify(input_data: TextInput):
    return handle_classification_request(input_data.content)

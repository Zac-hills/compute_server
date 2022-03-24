from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..models.translate import TranslateModelFactory
from ..models.classify import run_classification

router = APIRouter(prefix="/model")
model_factory = TranslateModelFactory()

class ClassifyPayload(BaseModel):
    texts: List[str]

class TranslatePayload(BaseModel):
    source: str
    to: str
    texts: List[str]

@router.post("/translate")
async def translate(payload: TranslatePayload):
    if not model_factory.supported(payload.source, payload.to):
        raise HTTPException(status_code=400, detail=model_factory.list_supported())
    model = model_factory.create_model(payload.source, payload.to)
    return {"translation": model.run(payload.text)}

@router.post("/classify")
async def classify(payload: ClassifyPayload):
    return {"classifications": run_classification(payload.texts)}
    
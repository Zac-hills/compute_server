from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..models.translate import TranslateModelFactory

router = APIRouter(prefix="/translate")
model_factory = TranslateModelFactory()

class TranslatePayload(BaseModel):
    source: str
    to: str
    text: List[str]

@router.post("/")
async def translate(payload: TranslatePayload):
    if not model_factory.supported(payload.source, payload.to):
        raise HTTPException(status_code=400, detail=model_factory.list_supported())
    model = model_factory.create_model(payload.source, payload.to)
    return {"translation": model.run(payload.text)}
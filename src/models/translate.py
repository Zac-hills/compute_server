
import json
import struct
from tarfile import SUPPORTED_TYPES
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TranslateModel:
    def __init__(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return
    def run(self, texts: List[str]) -> List[str]:
        result = []
        for text in texts:
            batch = self.tokenizer([text], return_tensors="pt")
            gen = self.model.generate(**batch)
            result.append(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return result

class TranslateModelFactory:
    SUPPORTED_TRANSLATIONS = {
        "fr->en" : "Helsinki-NLP/opus-mt-fr-en",
        "fi->en" : "Helsinki-NLP/opus-tatoeba-fi-en",
        "uk->en" : "Helsinki-NLP/opus-mt-uk-en",
    }

    def create_key(self, source: str, dest: str) -> str:
        return source.lower() + "->" + dest.lower()
    
    def supported(self, source: str, dest: str) -> bool:
        return self.create_key(source, dest) in self.SUPPORTED_TRANSLATIONS
    
    def create_model(self, source: str, dest: str) -> TranslateModel:
        if not self.supported(source, dest):
            return None
        return TranslateModel(self.SUPPORTED_TRANSLATIONS[self.create_key(source, dest)])
    
    def list_supported(self) -> str:
        supported_list = [{"source": k.split("->")[0], "to": k.split("->")[1]} for k,_ in self.SUPPORTED_TRANSLATIONS.items()]
        return json.dumps(supported_list)


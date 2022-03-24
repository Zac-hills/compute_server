from langid.langid import LanguageIdentifier, model
from typing import List

class ClassifyModel:
    def __init__(self) -> None:
        self.model = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    def run(self, texts: List[str]) -> List[str]:
        result = []
        for text in texts:
            result.append(self.model.classify(text))
        return result

def run_classification(texts: List[str]) -> List:
    model = ClassifyModel()
    return model.run(texts)
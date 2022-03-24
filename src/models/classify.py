import langid
from typing import List

class ClassifyModel:
    def run(self, texts: List[str]) -> List[str]:
        result = []
        for text in texts:
            result.append(langid.classify(text))
        return result

def run_classification(texts: List[str]) -> List:
    model = ClassifyModel()
    return model.run(texts)
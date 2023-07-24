from transformers import AutoTokenizer, AutoModelForMaskedLM


class LanguageModel:
    def __init__(self, model_name=None, model_path=None):
        if not model_name and not model_name:
            raise RuntimeError("can't load any model, because model_name and model_path are both None")

        path = model_name if model_path is None else model_path
        self.model = AutoModelForMaskedLM.from_pretrained(path)
        self.AutoTokenizer = AutoTokenizer.from_pretrained(path)

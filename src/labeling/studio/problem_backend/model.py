from label_studio_ml.model import LabelStudioMLBase
from transformers import AutoTokenizer, pipeline


class ProblemBackend(LabelStudioMLBase):

    def __init__(self, model_name='facebook/bart-large-mnli', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = pipeline(
            task="zero-shot-classification",
            model=model_name,
            tokenizer=self.tokenizer
        )

        self.from_name = "is_problem"
        self.to_name = "body"
        self.value = "choices"

    def predict(self, tasks, **kwargs):
        pass

    def fit(self, tasks, workdir=None, **kwargs):
        pass


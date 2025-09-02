
class AbstractModelTrainer(ABC):
    def __init__(
        self,
        config: dict,
        tokenizer,
        model,
        dataset,
        optimizer,
    ):
        pass
    def fit(self):
        pass
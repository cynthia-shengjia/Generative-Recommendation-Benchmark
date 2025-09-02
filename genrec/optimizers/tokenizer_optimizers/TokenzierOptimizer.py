import torch
from abc import ABC



class AbstractTokenizerOptimizer(ABC):
    def __init__(self, config: dict):
        self.config = config

    def _call_loss(self):
        raise NotImplementedError('Optimizer loss function not implemented.')

    def step(self):
        raise NotImplementedError('Optimizer step not implemented.')
    
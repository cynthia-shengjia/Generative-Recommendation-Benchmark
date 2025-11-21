# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from genrec.quantization.optimizers.base_optimizer import AbstractTokenizerOptimizer
from genrec.quantization.tokenizers.base_tokenizer import AbstractTokenizer
class Trainer:
    def __init__(
        self, 
        config: dict,  
        tokenizer: AbstractTokenizer,
        optimizer: AbstractTokenizerOptimizer
    ):
        self.config     = config
        self.tokenizer  = tokenizer 


    def fit(self, train_dataloader, val_dataloader):
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """
        raise NotImplementedError('Trainer fit not implemented.')




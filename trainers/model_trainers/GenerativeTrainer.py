import torch
from typing import Dict, List, Tuple
from transformers import T5ForConditionalGeneration,T5Config,Trainer
from genrec.cbs_structure.generate_trie import Trie, prefix_allowed_tokens_fn
class GenerativeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.generation_params = kwargs.pop('generation_params', {})
        self.item2tokens = kwargs.pop('item2tokens', None) # 传入tokenizer以便构建Trie
        self.pad_token_id = kwargs.pop('pad_token_id', 0)
        self.eos_token_id = kwargs.pop('eos_token_id', 1)
        super().__init__(*args, **kwargs)
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        else:
            self.prefix_allowed_fn = None
            print("警告: 未提供 tokenizer_for_gen 或 tokenizer.item2tokens，无法使用前缀约束。")


    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: List[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        重写 prediction_step 以执行生成操作。
        """
        # 如果是训练过程中的评估，则计算损失
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # 1. 计算常规损失
        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None

        # 2. 执行生成操作
        gen_kwargs = {
            # 对于T5，max_length 是指 decoder 生成序列的最大长度
            "max_length": self.generation_params.get('max_gen_length'), 
            "num_beams": self.generation_params.get('num_beams'),
            "num_return_sequences": self.generation_params.get('max_k'),
            "early_stopping": True,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
        }
        if self.prefix_allowed_fn:
            gen_kwargs["prefix_allowed_tokens_fn"] = self.prefix_allowed_fn

        generated_sequences = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # (batch_size * num_beams, seq_len) -> (batch_size, num_beams, seq_len)
        batch_size = inputs["input_ids"].shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        return (loss, generated_ids_reshaped, inputs.get("labels"))
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        
        # 1. 弹出自定义的 loss_mask
        #    **重要: 假设这个 mask 的形状是 [batch, seq_len, vocab_size]**
        loss_mask = inputs.pop("loss_mask", None)
        labels = inputs.get("labels")

        # 2. 正常运行模型前向传播
        outputs = model(**inputs)
        logits = outputs.logits # 形状: [batch, seq_len, vocab_size]

        loss = None
        # 3. 检查是否需要自定义 loss 计算
        if labels is not None:
            #loss_mask [batch_size, seq_len, vocab_size]
            if loss_mask is not None:
                
                masked_logits = logits.masked_fill(loss_mask == 0.0, -1e9)

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)


                loss = loss_fct(
                    masked_logits.view(-1, model.config.vocab_size), 
                    labels.view(-1)
                )
                
                
            else:
                loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
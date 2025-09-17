import torch
from typing import Dict, List, Optional
from transformers import Trainer
from torch.nn import CrossEntropyLoss
from genrec.cbs_structure.MSL_Tire import Trie, prefix_allowed_tokens_fn
class MSLTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.tau = kwargs.pop('tau', 1.0)  # 温度参数
        self.generation_params = kwargs.pop('generation_params', {})
        self.item2tokens = kwargs.pop('item2tokens', None)
        self.pad_token_id = kwargs.pop('pad_token_id', 0)
        self.eos_token_id = kwargs.pop('eos_token_id', 1)
        
        super().__init__(*args, **kwargs)
        
        # 初始化Trie用于约束生成
        if self.item2tokens:
            self.candidate_trie = Trie(self.item2tokens)
            self.prefix_allowed_fn = prefix_allowed_tokens_fn(self.candidate_trie)
        else:
            self.prefix_allowed_fn = None
            print("警告: 未提供 item2tokens，无法使用前缀约束。")

 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """重写损失计算，使用Trie约束的softmax（适用于Encoder-Decoder模型）"""
        # 获取约束掩码
        constrain_mask = inputs.pop("constrain_mask")
        
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]

        # 检查形状是否匹配
        batch_size, seq_len = labels.shape
        if logits.shape[0] != batch_size or logits.shape[1] != seq_len:
            logits = logits[:, :seq_len, :]
        
        # ===== 关键修改：直接去掉最后一个位置 =====
        # 移除最后一个位置（避免EOS位置的无效token问题）
        logits = logits[..., :-1, :]  # [batch_size, seq_len-1, vocab_size]
        labels = labels[..., :-1]     # [batch_size, seq_len-1]
        constrain_mask = constrain_mask[..., :-1, :]  # [batch_size, seq_len-1, vocab_size]
        # ========================================
        
        # 展平tokens
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_labels = labels.contiguous().view(-1)
        flat_constrain_mask = constrain_mask.contiguous().view(-1, constrain_mask.size(-1))
        
        # 确保labels在正确的设备上
        flat_labels = flat_labels.to(flat_logits.device)
        
        # 创建掩码，忽略填充token（-100）
        mask = flat_labels != -100
        
        # 应用掩码
        valid_labels = flat_labels[mask]
        valid_logits = flat_logits[mask]
        valid_constrain_mask = flat_constrain_mask[mask]
        
        # 应用约束：将无效token的logits设为负无穷
        valid_logits[valid_constrain_mask == 0] = -float("inf")
        
        """Trie约束的SSM损失"""
        # 正样本logits
        pos_logits = valid_logits.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
        pos_loss = -pos_logits / self.tau
        
        # 负样本logits
        neg_logits = torch.exp(valid_logits / self.tau)
        neg_sum = neg_logits.sum(dim=-1)        
        neg_loss = torch.log(neg_sum)
        
        # 计算最终损失
        loss = (pos_loss + neg_loss).mean()
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: List[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """重写预测步骤以支持约束生成"""
        # 计算常规损失
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
            else:
                loss = None
        
        # 执行约束生成
        gen_kwargs = {
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
        
        # 重塑生成结果
        batch_size = inputs["input_ids"].shape[0]
        num_return_sequences = gen_kwargs["num_return_sequences"]
        generated_ids_reshaped = generated_sequences.view(batch_size, num_return_sequences, -1)
        
        return (loss, generated_ids_reshaped, inputs.get("labels"))
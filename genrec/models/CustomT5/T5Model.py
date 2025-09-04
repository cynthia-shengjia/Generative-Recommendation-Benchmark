import torch
import torch.nn as nn
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from typing import Optional, Tuple, Union, List, Dict, Any
import copy
import math

class CustomT5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(
        self,
        config: T5Config,
        codebook_size: int,
        latent_dim: int = 384,  # 现在默认是 64 * 6 = 384
        tie_word_embeddings: bool = True
    ):
        # 初始化父类
        super().__init__(config)
        
        # 保存配置参数
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        
        # 创建嵌入层 - 大小应为 [codebook_size, latent_dim]
        self.shared = nn.Embedding(codebook_size, latent_dim)
        
        # 创建编码器和解码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        
        # 创建LM头 - 输出大小应为codebook_size
        self.lm_head = nn.Linear(config.d_model, codebook_size, bias=False)
        
        # 是否绑定权重
        if tie_word_embeddings:
            self.lm_head.weight = self.shared.weight
            
        # 初始化权重
        self.init_weights()

        
    def init_weights(self):
        """初始化模型权重"""
        # 初始化嵌入层
        nn.init.normal_(self.shared.weight, mean=0.0, std=0.02)
        
        # 初始化LM头
        if hasattr(self, 'lm_head'):
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
            
        # 初始化编码器和解码器
        self.encoder.init_weights()
        self.decoder.init_weights()
        
    def forward(
        self,
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Dict[str, torch.FloatTensor]]:
        """
        前向传播
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 使用编码器处理输入
        encoder_outputs = self.encoder(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 使用解码器生成输出
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 通过LM头获取logits
        sequence_output = decoder_outputs[0]
        logits = self.lm_head(sequence_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.codebook_size), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
            "encoder_last_hidden_state": encoder_outputs.last_hidden_state,
            "encoder_hidden_states": encoder_outputs.hidden_states,
            "encoder_attentions": encoder_outputs.attentions,
            "decoder_last_hidden_state": decoder_outputs.last_hidden_state,
            "decoder_hidden_states": decoder_outputs.hidden_states,
            "decoder_attentions": decoder_outputs.attentions,
        }
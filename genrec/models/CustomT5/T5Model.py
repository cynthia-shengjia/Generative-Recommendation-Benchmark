import torch
import torch.nn as nn
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union, List, Dict, Any
import copy
import math

class CustomT5ForConditionalGeneration(T5PreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: T5Config,
        vocab_size: int,
        latent_dim: int = 384,
        tie_word_embeddings: bool = True
    ):
        # 初始化父类
        super().__init__(config)
        
        # 保存配置参数
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        
        # 创建嵌入层
        self.shared = nn.Embedding(vocab_size, latent_dim)
        
        # 创建编码器和解码器
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)
        
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = True
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        
        # 创建LM头
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        
        # 是否绑定权重
        if tie_word_embeddings:
            self.lm_head.weight = self.shared.weight

        self.config.decoder_start_token_id = 0
        self.config.pad_token_id = 0
        self.config.eos_token_id = 1
        
        # 初始化权重
        self.init_weights()
        
    def init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.shared.weight, mean=0.0, std=0.02)
        
        if hasattr(self, 'lm_head'):
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
            
        self.encoder.init_weights()
        self.decoder.init_weights()
    
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past=None, 
        attention_mask=None, 
        use_cache=None, 
        encoder_outputs=None,
        **kwargs
    ):
        # 为生成准备输入
        if past is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_input_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        前向传播
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # 处理输入参数
        if encoder_input_ids is None and input_ids is not None:
            encoder_input_ids = input_ids
            
        if encoder_attention_mask is None and attention_mask is not None:
            encoder_attention_mask = attention_mask

        # 使用编码器处理输入（如果未提供编码器输出）
        if encoder_outputs is None:
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
            encoder_hidden_states=encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 通过LM头获取logits
        sequence_output = decoder_outputs[0] if not return_dict else decoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + (encoder_outputs,)
            return ((loss,) + output) if loss is not None else output

        # 使用Seq2SeqLMOutput而不是字典
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def _reorder_cache(self, past, beam_idx):
        """
        重新排序过去的键值缓存以匹配新的beam顺序
        """
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx) for past_state in layer_past
            ),)
        return reordered_past
from model.language_model import LLaDAModel,LLaDAConfig
import torch
from typing import List, Optional, Tuple, Union
from prune_token import PruneConfig
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
import random

@dataclass
class BaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None  
    retained_indices: Optional[torch.LongTensor] = None  # add by zichen 

logger = logging.get_logger(__name__)


class Random_v1(LLaDAModel):
    def __init__(self, config: LLaDAConfig):
        self.last_attention = None
        self.retained_indices = None
        self.prune_config = PruneConfig.instance()
        super().__init__(config)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,    
    ):
        #参数检查及初始化
        assert (past_key_values is None and not use_cache), "The kvcache is not suppotred for MDM."

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, is_causal=False) # Modify: MDM

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        #prune
        batch_size, seq_length = inputs_embeds.shape[:2]
        gen_length=self.prune_config.gen_lenghth #if self.config.gen_length is not None else 0
        suffix_length=self.prune_config.suffix_length #if self.config.suffix_len is not None else 0
        if self.prune_config.is_prune:
            num_block = self.prune_config.current_block
            num_step = self.prune_config.num_step
            K = self.prune_config.pruned_layer
            ratio=self.prune_config.reduction_ratio
            image_start=self.prune_config.image_token_start_index
            image_token_length=self.prune_config.image_token_length if self.config.text_length is None else (seq_length - self.config.text_length - gen_length - suffix_len)
            image_end = image_start + image_token_length
        else:
            K = -1

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                #prune核心代码
                if decoder_layer.self_attn.layer_idx == K and seq_length > 1 and num_block == 0 and num_step == 0:
                    device = hidden_states.device
                    image_indexs = random.sample(range(image_start, image_end), int((1 - ratio) * image_token_length))
                    keep_indexs = torch.cat((torch.arange(0, image_start, device=device), torch.Tensor(image_indexs).to(device), torch.arange(image_end, seq_length, device=device)))
                    keep_indexs = keep_indexs.sort().values.long()
                    # save keep_indexs
                    hidden_states = hidden_states[:,keep_indexs,:]#.contiguous()
                    cache_position = cache_position[keep_indexs]#.contiguous()  
                    causal_mask = self._update_causal_mask(
                            None, hidden_states, cache_position, is_causal=False
                        )#.contiguous()
                    position_ids = keep_indexs.unsqueeze(0)
                    self.retained_indices = position_ids.clone()
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hidden_states = self.norm(hidden_states)    
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)        
        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            retained_indices=self.retained_indices,
        )
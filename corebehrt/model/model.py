import torch.nn as nn
import numpy as np
from transformers import BertModel
from transformers.models.roformer.modeling_roformer import RoFormerEncoder
from torch.nn import CrossEntropyLoss, MSELoss
import torch

from embeddings.ehr import EhrEmbeddings
from model.activations import SwiGLU
from model.heads import FineTuneHead, MLMHead, BiGRU

"""
from transformers.activations import ACT2FN
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.roformer import *
from transformers.models.roformer.modeling_roformer import RoFormerSinusoidalPositionalEmbedding

from embeddings.ehr import EhrEmbeddings
from model.activations import SwiGLU
from model.heads import FineTuneHead, MLMHead
"""
"""
class RoFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size // config.num_attention_heads) 
        self.layer = nn.ModuleList([RoFormerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # [sequence_length, embed_size_per_head] -> [batch_size, num_heads, sequence_length, embed_size_per_head]
        sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1])[None, None, :, :]
        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            print('layer_module', layer_module)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            print('Hidden states', hidden_states.shape)
            print('attention_mask', attention_mask.shape)
            print('sinusoidal_pos', sinusoidal_pos.shape)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                print('Start layer_module')
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    sinusoidal_pos,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            print('Ended layer_module')
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
"""

class BertEHREncoder(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = EhrEmbeddings(config)
        
        # Activate transformer++ recipe
        config.rotary_value = False
        config.max_position_embeddings = 2048        
        self.encoder = RoFormerEncoder(config)
        for layer in self.encoder.layer:
            layer.intermediate.intermediate_act_fn = SwiGLU(config)
        
    def forward(self, batch: dict):
        position_ids = {key: batch[key] for key in ['age', 'abspos']} # abspos
        outputs = super().forward(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['segment'],
            position_ids=position_ids,
        )
        """
        # Call the custom embeddings directly
        embedding_output = self.embeddings(input_ids=batch['concept'], token_type_ids=batch['segment'],
                                           position_ids = position_ids)
        print('in-function end embedding')
        encoder_outputs = self.encoder(embedding_output, attention_mask=batch['attention_mask'])
        print('in-function end encoder')
        outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if self.config.output_hidden_states else None,
            attentions=encoder_outputs[2] if self.config.output_attentions else None)
        
        """
        return outputs

class BertEHRModel(BertEHREncoder):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = nn.CrossEntropyLoss()
        self.cls = MLMHead(config)
            
    def forward(self, batch: dict):
        outputs = super().forward(batch)
        sequence_output = outputs[0]    # Last hidden state
        logits = self.cls(sequence_output) # CHANGE , batch['attention_mask']
        outputs.logits = logits

        if batch.get('target') is not None:
            outputs.loss = self.get_loss(logits, batch['target'])

        return outputs

    def get_loss(self, logits, labels):
        """Calculate loss for masked language model."""
        return self.loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))


class BertForFineTuning(BertEHRModel):
    def __init__(self, config):
        super().__init__(config)

        self.loss_fct = nn.BCEWithLogitsLoss()
        self.cls = FineTuneHead(config)
        
    def get_loss(self, hidden_states, labels):    
        return self.loss_fct(hidden_states.view(-1), labels.view(-1))


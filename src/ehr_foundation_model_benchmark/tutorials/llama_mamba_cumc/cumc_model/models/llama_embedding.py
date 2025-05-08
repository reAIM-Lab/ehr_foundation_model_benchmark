import torch
from transformers import AutoConfig, AutoModelForCausalLM
from typing import Dict, List, Any, Optional, Union
from omegaconf import DictConfig
from jaxtyping import Float
from typing import Dict, Any, Optional
import pickle
import os,sys

from hf_ehr.models.base import BaseModel

'''
Llama: combine both pretrained embedding and fone embeddings
'''

'''
( input_ids: typing.Optional[torch.LongTensor] = None   attention_mask: typing.Optional[torch.Tensor] = None   position_ids: typing.Optional[torch.LongTensor] = None,
past_key_values: typing.Optional[transformers.cache_utils.Cache] = Noneinputs_embeds: typing.Optional[torch.FloatTensor] = None,
labels: typing.Optional[torch.LongTensor] = None use_cache: typing.Optional[bool] = None output_attentions: typing.Optional[bool] = None,
output_hidden_states: typing.Optional[bool] = None  cache_position: typing.Optional[torch.LongTensor] = None  logits_to_keep: typing.Union[int, torch.Tensor] = 0,
**kwargs: typing_extensions.Unpack[transformers.models.llama.modeling_llama.KwargsForCausalLM] ) → transformers.modeling_outputs.CausalLMOutputWithPast or tuple(torch.FloatTensor)
'''

class LlamaLanguageModel(BaseModel):
    """
    Llama with a Language Model head.
    """

    def __init__(self, config: DictConfig, vocab_size, pad_token_id,pretrained_embedding_path=None) -> None:
        super(LlamaLanguageModel, self).__init__(config, vocab_size, pad_token_id)
        
        # Enable flash attention
        if torch.cuda.get_device_capability('cuda')[0] >= 8:
            kwargs = {
                'attn_implementation': 'flash_attention_2',
                'torch_dtype': torch.bfloat16,
            }
        else:
            kwargs = {}

        # Model specs
        print(f"config.model.hf_name is {config.model.hf_name}")
        model_config = AutoConfig.from_pretrained(config.model.hf_name, trust_remote_code=True, use_cache=False, **kwargs)
        model_config.vocab_size = vocab_size
        print(f"model_config is {model_config}")
        for key, val in config.model.config_kwargs.items():
            assert hasattr(model_config, key), f"Config for HF model {config.model.hf_name if hasattr(config.model, 'hf_name') else ''} does not have attribute {key}"
            setattr(model_config, key, val)
        self.model_config = model_config
        self.hidden_size = model_config.hidden_size

        # Model
        self.model = AutoModelForCausalLM.from_config(model_config, **kwargs)

        self.pretrained_embeddings = self.load_pretrained_embedding()

        # if os.path.exists(pretrained_embedding_path):
        #     self.pretrained_embeddings = self.load_pretrained_embedding()
        # else:
        #     print(f"error, the specified path doesn't exist in {pretrained_embedding_path}")
        #     sys.exit(1)

        # Run any post-init handlers from super()
        self.post_init()

        ## To do 
        # add fone_module
        # add embedding_projection 

    # def value_embedding()
        
        
    def load_pretrained_embedding(self):
        try:
            with open(self.pretrained_embedding_path,'rb') as f:
                embeddings = pickle.load(f)
        except Exception as e:
            print(f"Failed to load pretrained embeddings: {e}")
        return embeddings
    
    # they arrange samples in order before collate
    def create_custom_embeddings(self, original_events, input_ids, attention_mask):
        '''
        combine
        '''

        # initialize input embeddings
        batch_size, seq_length = input_ids.size()
        inputs_embeds = torch.zeros((batch_size, seq_length, self.hidden_size),device=input_ids.device)

        # process each batch item
        for batch_idx, events in enumerate(batch_size):
            # Create mapping between sequence positions and events
            seq_to_event = {}
            for seq_idx, event in enumerate(events):
                if seq_idx < seq_length:  # Ensure we don't exceed sequence length
                    seq_to_event[seq_idx] = event
            
            # Fill in embeddings
            for seq_idx in range(seq_length):
                if seq_idx in seq_to_event and attention_mask[batch_idx, seq_idx] == 1:
                    event = seq_to_event[seq_idx]
                    # Get code embedding
                    code_embedding = self.get_code_embedding(event.code)
                    
                    # If numerical value exists, generate value embedding
                    if event.value is not None and isinstance(event.value, (int, float)):
                        value_tensor = torch.tensor([[event.value]], device=code_embedding.device)
                        value_embedding = self.fone_module(value_tensor)
                        # Combine code and value embeddings
                        combined = torch.cat([code_embedding, value_embedding.squeeze(0)], dim=0)
                        embedding = self.embedding_projection(combined.unsqueeze(0)).squeeze(0)
                    else:
                        # For events without numerical values, just use code embedding
                        embedding = code_embedding
                    
                    inputs_embeds[batch_idx, seq_idx] = embedding
                else:
                    # For padding positions or positions beyond available events
                    # Use the embedding from the input_ids
                    if attention_mask[batch_idx, seq_idx] == 0:
                        # This is a padding position
                        inputs_embeds[batch_idx, seq_idx] = self.original_word_embeddings(
                            torch.tensor([self.pad_token_id], device=input_ids.device)
                        ).squeeze(0)
                    else:
                        # This is a position where we don't have event data but need an embedding
                        # Use the standard embedding from the model
                        inputs_embeds[batch_idx, seq_idx] = self.original_word_embeddings(
                            input_ids[batch_idx, seq_idx].unsqueeze(0)
                        ).squeeze(0)
        
        return inputs_embeds

    #batch is returned value from collate_femr_timelines
    def training_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        tokens.pop("token_type_ids", None)

        
        # replace idx inputs with embedding inputs outputs = self.model(**tokens)

        # load tokenized results
        input_ids = tokens["input_ids"]
        attention_mask = tokens['attention_mask']
        labels = tokens.get("labels")
        values = tokens.get('values')

        # Remove token_type_ids if present
        tokens.pop("token_type_ids", None)


        # create input embeddings based on codes & values
        input_embds = self.create_custom_embeddings(input_ids,attention_mask,values)

        outputs = self.model(
            input_ids=None, 
            attention_mask = attention_mask,
            input_embeds=input_embds,
            labels=labels,
            return_dict=True
        )

        loss: torch.Tensor = outputs.loss
        
        # Learning rate scheduler
        lr: float = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]["lr"]
        sch = self.lr_schedulers()
        sch.step()
        
        # Logging + Metrics
        self.log_training_step(loss.detach(), B, tokens, lr)

        return loss
    

    def validation_step(self, 
                      batch: Dict[str, Any],
                      batch_idx: int) -> Optional[torch.Tensor]:
        tokens: Dict[str, Float[torch.Tensor, 'B L']] = batch['tokens']
        B: int = tokens['input_ids'].shape[0]

        tokens.pop("token_type_ids", None)

        # load tokenized results
        input_ids = tokens["input_ids"]
        attention_mask = tokens['attention_mask']
        labels = tokens.get("labels")

        # load tokenized batches
        original_events = batch.get('original_events')

        input_embds = self.create_custom_embeddings(original_events,input_ids,attention_mask)
        outputs = self.model(
            input_ids=None, 
            attention_mask = attention_mask,
            input_embeds=input_embds,
            labels=labels,
            return_dict=True
        )

        loss: torch.Tensor = outputs.loss

        # Handle NaN checks from your original code
        # actually I don't quite understand 
        if torch.isnan(loss).any():
            nan_detected = torch.tensor([1.0], device=self.device)
        else:
            nan_detected = torch.tensor([0.0], device=self.device)

        torch.distributed.all_reduce(nan_detected, op=torch.distributed.ReduceOp.MAX)
        if nan_detected.item() == 1:
            print("NaN detected in loss, skipping this batch across all processes.")
            return None
        
        self.log_validation_step(loss, tokens)
        return loss

if __name__ == '__main__':
    model = LlamaLanguageModel()
    
    outputs = model.model(**tokens)
    model.model.backward(outputs.loss)

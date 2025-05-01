"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""
import os
from os.path import join

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

import torch
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
from datasets import Dataset

from common.config import load_config, instantiate
from common.initialize import Initializer
from common.utils import compute_number_of_warmup_steps
from trainer.copy_trainer import CustomTrainer, CustomEarlyStoppingCallback, LossLoggingCallback
from common.loader import ModelLoader, load_model_cfg_from_checkpoint
from model.model import BertEHRModel

print('Checking flash attention', torch.backends.cuda.flash_sdp_enabled())
print('Checking memory efficient sdp',torch.backends.cuda.mem_efficient_sdp_enabled())
print('Checking math sdp enabled',torch.backends.cuda.math_sdp_enabled())
torch.set_float32_matmul_precision("high")
device = 'cuda:0'
config_path = 'configs/pretrain.yaml'

def prepare_mlm_labels(data_dict):
    """
    Expects a dict with 'input_ids' and 'attention_mask'.
    Assumes masked tokens are already in 'input_ids' as [MASK] tokens (usually token ID 103 for BERT).
    Sets unmasked token positions in 'labels' to -100 so they are ignored in the loss.
    """
    def mask_non_masked_tokens(example):
        mask_token_id = 4  # typical [MASK] for BERT is 123, in vocab it is 4
        input_ids = example['input_ids']
        labels = input_ids.copy()

        # Replace unmasked tokens with -100
        labels = [label if label == mask_token_id else -100 for label in labels]
        example['labels'] = labels
        return example

    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.map(mask_non_masked_tokens, batched=False)
    return dataset

def initialize_pretrain_model(cfg, vocab_size):
    """Initialize model from checkpoint or from scratch."""
    print('Initializing new model')
    model = BertEHRModel(BertConfig(**cfg.model, vocab_size=vocab_size, attn_implementation="flash_attention_2"))
    print('Checking flash attention', model.config.attn_implementation)
    return model

def main_train(config_path):
    cfg = load_config(config_path)
    
    epochs = 20
    warmup_epochs = 5
    batchsize = 32
    
    path_to_tokenized = '/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_corebehrt/corebehrt_data/tokenized/'
    output_path= '/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_corebehrt/corebehrt_data/training/'
    
    vocab = torch.load(path_to_tokenized + 'vocabulary.pt', weights_only=False)
    """
    train_dataset = torch.load(path_to_tokenized + 'tokenized_pretrain.pt', weights_only = False)
    hf_train_dataset = {"input_ids" if k == 'concept' else k:v for k,v in train_dataset.items()}
    hf_train_dataset = prepare_mlm_labels(hf_train_dataset)
    """

    val_dataset = torch.load(path_to_tokenized + 'tokenized_finetune.pt', weights_only = False)
    hf_val_dataset = {"input_ids" if k == 'concept' else k:v for k,v in val_dataset.items()}

    vocab_size = len(vocab)
    print('Vocab size:', vocab_size)
    model = initialize_pretrain_model(cfg, vocab_size)
    optimizer = AdamW(model.parameters(), lr=0.001, eps=1e-06, weight_decay=0.01)
    
    num_warmup_steps = warmup_epochs * len(hf_train_dataset) // batchsize
    num_training_steps = (epochs-warmup_epochs) * len(hf_train_dataset) // batchsize
    print('Scheduler step counts', num_warmup_steps, num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    model_parameters = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {model_parameters:,}')
        
    print('Initialize trainer')
    
    training_args = TrainingArguments(output_dir=output_path,
                                      evaluation_strategy = 'steps',
                                      eval_steps = 500,
                                      per_device_train_batch_size = batchsize,
                                      per_device_eval_batch_size = batchsize,
                                      learning_rate = 0.001,
                                      weight_decay = 0.01,
                                      num_train_epochs = 10,
                                      data_seed = 31,
                                      seed = 31,
                                      disable_tqdm = False,
                                      metric_for_best_model = 'eval_loss',
                                     load_best_model_at_end = True)
                                      
    trainer = Trainer(model = model, 
       args = training_args,
       train_dataset = hf_train_dataset,
       eval_dataset = hf_val_dataset,
       callbacks = [CustomEarlyStoppingCallback(1, 0.01),
                   LossLoggingCallback(log_file=os.path.join(output_path, "loss_log.json"))],
       data_collator = dynamic_padding,
       optimizers = (optimizer, scheduler))


    print('Start training.train')
    trainer.train()
    logger.info("Done")


if __name__ == '__main__':
    main_train(config_path)

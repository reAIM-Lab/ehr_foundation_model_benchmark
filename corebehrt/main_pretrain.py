"""Pretrain BERT model on EHR data. Use config_template pretrain.yaml. Run main_data_pretrain.py first to create the dataset and vocabulary."""
import os
from os.path import join
import torch
from transformers import Trainer, TrainingArguments
from datasets import Dataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

from common.azure import AzurePathContext, save_to_blobstore
from common.config import load_config, instantiate
from common.initialize import Initializer
from common.loader import (load_checkpoint_and_epoch,
                                   load_model_cfg_from_checkpoint)
from common.setup import DirectoryPreparer, copy_data_config, get_args
from common.utils import compute_number_of_warmup_steps
from data.prepare_data import DatasetPreparer
from dataloader.collate_fn import dynamic_padding
from trainer.trainer import EHRTrainer
from trainer.copy_trainer import CustomTrainer, CustomEarlyStoppingCallback, LossLoggingCallback

CONFIG_NAME = 'pretrain.yaml'
BLOBSTORE = 'PHAIR'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

print('Checking flash attention', torch.backends.cuda.flash_sdp_enabled())
print('Checking memory efficient sdp',torch.backends.cuda.mem_efficient_sdp_enabled())
print('Checking math sdp enabled',torch.backends.cuda.math_sdp_enabled())
torch.set_float32_matmul_precision("high")


def main_train(config_path):
    cfg = load_config(config_path)

    cfg, run, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).adjust_paths_for_azure_pretrain()

    logger, run_folder = DirectoryPreparer.setup_run_folder(cfg)
    copy_data_config(cfg, run_folder)
    
    loaded_from_checkpoint = load_model_cfg_from_checkpoint(cfg, 'pretrain_config.yaml') # if we are training from checkpoint, we need to load the old config
    train_dataset, val_dataset = DatasetPreparer(cfg).prepare_mlm_dataset()

    hf_train_dataset = {"input_ids" if k == 'concept' else k:v for k,v in train_dataset.features.items()}
    hf_train_dataset = Dataset.from_dict(hf_train_dataset)
    hf_train_dataset.reset_format()
    hf_train_dataset.set_format(type='torch', columns=['input_ids', 'age', 'segment', 'abspos', 'attention_mask'])

    print('train data length hf', len(hf_train_dataset))
    print('train data type', type(hf_train_dataset))
    print("Underlying table size:", hf_train_dataset._data.num_rows)
    print("Indices size:", len(hf_train_dataset._indices) if hf_train_dataset._indices else "None")

    hf_val_dataset = {"input_ids" if k == 'concept' else k:v for k,v in val_dataset.features.items()}
    hf_val_dataset = Dataset.from_dict(hf_val_dataset)
    hf_val_dataset.reset_format()
    hf_val_dataset.set_format(type='torch', columns=['input_ids', 'age', 'segment', 'abspos', 'attention_mask'])

    
    if 'scheduler' in cfg:
        logger.info('Computing number of warmup steps')
        compute_number_of_warmup_steps(cfg, len(train_dataset))
        print('After warmup', [key for key in cfg.scheduler])

    checkpoint, epoch = load_checkpoint_and_epoch(cfg)
    logger.info(f'Continue training from epoch {epoch}')    
    initializer = Initializer(cfg, checkpoint=checkpoint)
    model = initializer.initialize_pretrain_model(train_dataset)
    logger.info('Initializing optimizer')
    optimizer = initializer.initialize_optimizer(model)
    scheduler = initializer.initialize_scheduler(optimizer)

    model_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f'Total parameters: {model_parameters:,}')

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Trainable parameters: {trainable_params:,}')
        
    logger.info('Initialize trainer')
    
    training_args = TrainingArguments(output_dir=cfg.paths.output_path,
                                      evaluation_strategy = 'steps',
                                      eval_steps = 500,
                                      per_device_train_batch_size = 32,
                                      per_device_eval_batch_size = 32,
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
                   LossLoggingCallback(log_file=os.path.join(cfg.paths.output_path, "loss_log.json"))],
       data_collator = dynamic_padding,
       optimizers = (optimizer, scheduler))

    """
    trainer = EHRTrainer( 
        model=model, 
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        cfg=cfg,
        logger=logger,
        run=run,
        last_epoch=epoch
    )
    logger.info('Start training')
    trainer.train()
    
    compute_metrics = {k: instantiate(v) for k, v in cfg.metrics.items()}
    trainer = CustomTrainer(model = model,
                            optimizer = optimizer,
                            lr_scheduler = scheduler, 
                            train_dataset = train_dataset,
                            eval_dataset = val_dataset,
                            compute_metrics = compute_metrics,
                            **cfg.trainer_args)
    """
    print('Start training.train')
    trainer.train()
    logger.info("Done")


if __name__ == '__main__':
    main_train(config_path)

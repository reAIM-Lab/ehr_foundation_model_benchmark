from transformers import Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import Dataset
import os
import logging
import numpy as np
import torch
from typing import Dict, Any
import json


from torch.utils.data import DataLoader, Dataset
from common.config import Config, instantiate
from dataloader.collate_fn import dynamic_padding
from trainer.utils import (compute_avg_metrics,
                                   get_tqdm)



class CustomEarlyStoppingCallback(EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        print(abs(metric_value - state.best_metric) / state.best_metric)
        self.early_stopping_threshold = 0.01
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) / state.best_metric > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            
            
class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_file="loss_log.json"):
        self.log_file = log_file
        self.logs = []

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        self.logs.append({"step": state.global_step, **logs})
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)

            

class CustomTrainer(Trainer):
    def __init__(
        self,
        model,
        train_dataset: Dataset = None,
        eval_dataset: Dataset = None,
        test_dataset: Dataset = None,
        compute_metrics: callable = None,
        output_dir: str = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_corebehrt/pretraining/checkpoints",
        logging_dir: str = '/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_corebehrt/pretraining/logging',
        logging_steps: int = 100,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        save_strategy: str = "epoch",
        save_total_limit: int = 3,
        eval_strategy: str = "epoch",
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        num_train_epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        fp16: bool = False,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        run_name: str = 'pretrain',
        **kwargs):
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir or os.path.join(output_dir, "logs"),
            logging_steps=logging_steps,
            evaluation_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            fp16=fp16,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            run_name=run_name,
        )
        
        self._custom_optimizer = optimizer
        self._custom_scheduler = lr_scheduler

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        
        self.add_callback(CustomEarlyStoppingCallback())


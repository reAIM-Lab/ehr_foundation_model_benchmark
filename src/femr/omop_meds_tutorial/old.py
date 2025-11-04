
import dataclasses
from typing import Optional, Tuple
from transformers import TrainingArguments, HfArgumentParser

import numpy as np
import transformers
import pathlib
import torch
import sys
import femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor


class CustomEarlyStoppingCallback(transformers.EarlyStoppingCallback):
    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric) / state.best_metric
                > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1


@dataclasses.dataclass
class MotorArguments:
    pretraining_data: str = dataclasses.field(
        metadata={
            "help": "Pretraining data folder"
        },
    )
    meds_reader: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "The folder for the meds reader"
        },
    )
    checkpoint_dir: Optional[str] = dataclasses.field(
        default=None,
        metadata={
            "help": "The checkpoint dir to restore the training from"
        }
    )
    n_layers: int = dataclasses.field(
        default=11,
        metadata={
            "help": "Pretraining data folder"
        },
    )


def parse_arguments()-> (
    Tuple[MotorArguments, TrainingArguments]
):
    parser = HfArgumentParser((MotorArguments, TrainingArguments))
    motor_args, training_args = parser.parse_args_into_dataclasses()
    return motor_args, training_args


def main():
    motor_args, training_args = parse_arguments()
    pretraining_data = pathlib.Path(motor_args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        tokenizer_path, ontology=ontology
    )

    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)

    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        n_layers=motor_args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        motor_task.get_task_config()
    )

    model = femr.models.transformer.FEMRModel(config)
    # model = model.to(torch.device("cuda"))
    #
    # learning_rate = args.learning_rate
    # output_dir = 'tmp_trainer_' + sys.argv[1]
    # trainer_config = transformers.TrainingArguments(
    #     per_device_train_batch_size=args.per_device_train_batch_size,
    #     per_device_eval_batch_size=args.per_device_eval_batch_size,
    #
    #     learning_rate=learning_rate,
    #     output_dir=output_dir,
    #     remove_unused_columns=False,
    #     bf16=True,
    #
    #     weight_decay=0.1,
    #     adam_beta2=0.95,
    #
    #     report_to=["tensorboard"],
    #
    #     num_train_epochs=args.n_epochs,
    #
    #     warmup_steps=500,
    #
    #     logging_strategy='epoch',
    #     logging_steps=10,
    #
    #     save_strategy='epoch',
    #     evaluation_strategy='epoch',
    #
    #     # prediction_loss_only=True,
    #     dataloader_num_workers=12,
    #
    #     save_total_limit=10,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    # )

    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=training_args,
        callbacks=[CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001)],
    )
    train_result = trainer.train(resume_from_checkpoint=motor_args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
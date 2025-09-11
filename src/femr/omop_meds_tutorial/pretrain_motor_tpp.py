import numpy as np
import transformers
import pathlib
import torch
import sys
import femr.models_tpp.transformer
import femr.models_tpp.config
import pickle
import datasets
import femr.models_tpp.tokenizer
import femr.models_tpp.processor
from femr.omop_meds_tutorial.motor_evaluation.generate_labels import create_omop_meds_tutorial_arg_parser
import torch.nn as nn
from transformers import TrainerCallback
import wandb

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


def create_arg_parser():
    arg_parser = create_omop_meds_tutorial_arg_parser()
    arg_parser.add_argument(
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default=None
    )

    arg_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=None
    )
    arg_parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-5
    )
    arg_parser.add_argument(
        "--n_layers",
        dest="n_layers",
        type=int,
        default=11
    )
    arg_parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=50
    )
    arg_parser.add_argument(
        "--per_device_train_batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        "--per_device_eval_batch_size",
        dest="per_device_eval_batch_size",
        type=int,
        default=1
    )
    arg_parser.add_argument(
        "--linear_interpolation",
        dest="linear_interpolation",
        type=bool,
        default=False
    )
    return arg_parser

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class WandbTrainLossCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        log_data = {}
        
        # Train loss vs step
        if "loss" in logs:
            log_data["train/loss_step"] = logs["loss"]
        
        # Eval loss vs step
        if "eval_loss" in logs:
            log_data["eval/loss_step"] = logs["eval_loss"]
        
        # Add epoch-based logging
        if "loss" in logs:
            log_data["train/loss_epoch"] = logs["loss"]
        if "eval_loss" in logs:
            log_data["eval/loss_epoch"] = logs["eval_loss"]
        
        # Record step and epoch for W&B axes
        log_data["global_step"] = state.global_step
        log_data["epoch"] = state.epoch
        
        wandb.log(log_data)


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)

    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models_tpp.tokenizer.HierarchicalTokenizer.from_pretrained(
        tokenizer_path, ontology=ontology
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)
    print(f"Motor task: {motor_task}")
    print(f"Motor task length: {len(motor_task.pretraining_task_codes)}")
    processor = femr.models_tpp.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))
    print(f"Train batches length: {len(train_batches)}, batch : {train_batches}")

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    # Finally, given the batches, we can train CLMBR.
    # We can use huggingface's trainer to do this.
    transformer_config = femr.models_tpp.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models_tpp.tokenizer.HierarchicalTokenizer),
        n_layers=args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    config = femr.models_tpp.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        motor_task.get_task_config()
    )

    # print(f"Transformer config: {transformer_config}")
    model = femr.models_tpp.transformer.FEMRModel(config,attn_implementation="flash_attention_2",linear_interpolation=args.linear_interpolation)
    model = model.to(torch.device("cuda:0"))


    print(f"Model param count: {count_parameters(model)}")

    learning_rate = args.learning_rate
    # output_dir = 'tmp_trainer_' + sys.argv[1]
    output_dir = args.output_dir
    # print(args.per_device_train_batch_size)
    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        learning_rate=learning_rate,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=True,

        weight_decay=0.1,
        adam_beta2=0.95,
        # report_to="none",
        report_to=["wandb"],
        run_name="mtpp_mimic_bin_8",
        # run_name="motor_pretrain_mimic",
        num_train_epochs=args.n_epochs,
        ddp_find_unused_parameters=False,

        warmup_steps=500,

        logging_strategy='epoch',
        # logging_strategy='steps',
        logging_steps=10,

        save_strategy='epoch',
        eval_strategy='epoch',
        # evaluation_strategy='epoch',

        # prediction_loss_only=True,
        # dataloader_num_workers=1,
        dataloader_num_workers=64,

        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    print(f"CONFIRMATION: The actual per_device_train_batch_size is {trainer_config.per_device_train_batch_size}")



    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
        callbacks=[CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001)],
    )

    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


# python -u -m femr.omop_meds_tutorial.pretrain_motor \
#   --pretraining_data $PRETRAINING_DATA \
#   --meds_reader $OMOP_MEDS_READER

'''
40 hours
export CUDA_VISIBLE_DEVICES=3

python pretrain_motor_tpp.py \
  --pretraining_data /user/zj2398/cache/deephit_tpp_8k \
  --meds_reader /user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/deephit_tpp_8k/output

  17.5

gsb
CUDA_VISIBLE_DEVICES=1,3,5 accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  --gpu_ids "1,3,5" \
  pretrain_motor_tpp.py \
  --pretraining_data /user/zj2398/cache/deephit_tpp_8k \
  --meds_reader /user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/deephit_tpp_8k/output

CUDA_VISIBLE_DEVICES=1 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --gpu_ids "1" \
  pretrain_motor_tpp.py \
  --pretraining_data /user/zj2398/cache/deephit_tpp_8k \
  --meds_reader /user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/deephit_tpp_8k/output

kuvira
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --gpu_ids "0" \
  pretrain_motor.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  --per_device_train_batch_size 1 \
  --output_dir /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8/output_exlude_corrected 


CUDA_VISIBLE_DEVICES=3 accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  --gpu_ids "3" \
  pretrain_motor.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  --per_device_train_batch_size 1 \
  --output_dir /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8/output_linear_interpolation \
  --linear_interpolation True
'''
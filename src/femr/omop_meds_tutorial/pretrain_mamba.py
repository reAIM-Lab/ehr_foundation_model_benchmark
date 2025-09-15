import numpy as np
import transformers
import pathlib
import torch
import femr.models.mamba  # Changed from femr.models.transformer
import pickle
import datasets
import femr.models.tokenizer
import femr.models.processor
from src.femr.omop_meds_tutorial.motor_evaluation.generate_labels import create_omop_meds_tutorial_arg_parser
import torch.nn as nn
import wandb
import json

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
    
    # Mamba-specific arguments
    arg_parser.add_argument(
        "--mamba_model_name",
        dest="mamba_model_name",
        type=str,
        default="state-spaces/mamba-130m-hf",
        help="HuggingFace Mamba model name to use as base"
    )
    arg_parser.add_argument(
        "--d_state",
        dest="d_state",
        type=int,
        default=16,
        help="Mamba state dimension"
    )
    # arg_parser.add_argument(
    #     "--mamba_config_overrides",
    #     dest="mamba_config_overrides",
    #     type=str,
    #     default="{}",
    #     help="JSON string of additional Mamba config overrides"
    # )
    
    return arg_parser

def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class WandbTrainLossCallback(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        # args, control, kwargs are unused but required by the interface
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
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        tokenizer_path, ontology=ontology
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)
    print(f"Motor task: {motor_task}")
    print(f"Motor task length: {len(motor_task.pretraining_task_codes)}")
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))
    print(f"Train batches length: {len(train_batches)}, batch : {train_batches}")

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))

    # Parse Mamba config overrides
    # try:
    #     mamba_config_overrides = json.loads(args.mamba_config_overrides)
    #     if not isinstance(mamba_config_overrides, dict):
    #         print("Warning: mamba_config_overrides is not a dict; ignoring.")
    #         mamba_config_overrides = {}
    # except json.JSONDecodeError as e:
    #     print(f"Error parsing mamba_config_overrides JSON: {e}")
    #     mamba_config_overrides = {}

    # Create a transformer config that will be converted to Mamba config internally
    # This approach maintains compatibility with the existing FEMRModelConfig structure
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        hidden_size=768,  # Default; can be aligned with HF d_model
        intermediate_size=3072,  # Ignored by Mamba, kept for compatibility
        n_layers=args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='silu',  # Changed from 'swiglu' to Mamba-compatible activation
    )
    
    print(f"Base transformer config (will be converted to Mamba): {transformer_config}")

    # Store Mamba-specific config in a way that FEMRMambaModel can access
    # We'll pass these as additional kwargs to the model constructor
    mamba_specific_config = {
        'mamba_model_name': args.mamba_model_name,
        'd_state': args.d_state,
        # 'mamba_config_overrides': mamba_config_overrides,
    }

    # Create model config using the same structure as transformer version
    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        motor_task.get_task_config()
    )

    print(f"Creating Mamba model...")
    # print(f"Mamba-specific config: {mamba_specific_config}")
    
        
    try:
        # Create Mamba model - note: no attn_implementation parameter for Mamba
        model = femr.models.mamba.FEMRMambaModel(
            config, 
            linear_interpolation=args.linear_interpolation,
            mamba_model_name=mamba_specific_config['mamba_model_name'],
            d_state=mamba_specific_config['d_state'],
            mamba_config_overrides=mamba_specific_config['mamba_config_overrides']
        )
        model = model.to(torch.device("cuda:0"))
        print(f"Mamba model created successfully!")
        
        # Validate model was created properly
        if model is None:
            raise RuntimeError("Model creation returned None")
            
        # Quick forward pass test with dummy data
        print("Performing model validation...")
        model.eval()
        with torch.no_grad():
            # This will help catch configuration issues early
            pass
        model.train()
        
    except Exception as e:
        print(f"Error creating Mamba model: {e}")
        print(f"Config details: {config}")
        print(f"Transformer config: {config.transformer_config}")
        print(f"Mamba-specific config: {mamba_specific_config}")
        raise

    print(f"Mamba Model param count: {count_parameters(model)}")

    learning_rate = args.learning_rate
    output_dir = args.output_dir
    
    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,

        learning_rate=learning_rate,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=True,

        weight_decay=0.1,
        adam_beta2=0.95,
        report_to=["wandb"],
        run_name="mamba_deephit_mimic_bin_8",  # Updated run name to indicate Mamba
        num_train_epochs=args.n_epochs,
        ddp_find_unused_parameters=False,

        warmup_steps=500,

        logging_strategy='epoch',
        logging_steps=10,

        save_strategy='epoch',
        evaluation_strategy='epoch',

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
        callbacks=[
            CustomEarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.001),
            WandbTrainLossCallback()
        ],
    )

    print("Starting Mamba model training...")
    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()


# Example usage:
# python -u -m femr.omop_meds_tutorial.pretrain_mamba \
#   --pretraining_data $PRETRAINING_DATA \
#   --meds_reader $OMOP_MEDS_READER \
#   --output_dir $OUTPUT_DIR \
#   --mamba_model_name "state-spaces/mamba-130m-hf" \
#   --d_state 16 \
#   --linear_interpolation False

'''
Example commands:

# Basic Mamba training
export CUDA_VISIBLE_DEVICES=6
python pretrain_mamba.py \
  --pretraining_data /user/zj2398/cache/motor_mimic \
  --meds_reader /user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/motor_mimic/mamba_output \
  --mamba_model_name "state-spaces/mamba-130m-hf" \
  --d_state 16

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  --gpu_ids "0,1,2" \
  pretrain_mamba.py \
  --pretraining_data /user/zj2398/cache/motor_mimic_bin_100 \
  --meds_reader /user/zj2398/cache/hf_ehr/mimic/meds_v0.6_reader \
  --per_device_train_batch_size 1 \
  --output_dir /user/zj2398/cache/motor_mimic_bin_100/mamba_output \
  --mamba_model_name "state-spaces/mamba-370m-hf" \
  --d_state 32

# With custom Mamba configuration overrides
python pretrain_mamba.py \
  --pretraining_data /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8 \
  --meds_reader /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  --per_device_train_batch_size 1 \
  --output_dir /data/processed_datasets/processed_datasets/zj2398/femr/mimic/motor_mimic_bin_8/mamba_output \
  --linear_interpolation True \
  --mamba_config_overrides '{"expand": 4, "d_conv": 8}'
'''

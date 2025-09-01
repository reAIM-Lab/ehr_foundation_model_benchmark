import numpy as np
import transformers
import pathlib
import torch
import sys
import pickle
import datasets
import torch.nn as nn
from transformers import TrainerCallback
import wandb

import femr.models.tokenizer
import femr.models.processor
import femr.transformer_v2 as femr_transformer  # Use the new transformer module
from src.femr.omop_meds_tutorial.motor_evaluation.generate_labels import create_omop_meds_tutorial_arg_parser


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
    """Enhanced argument parser with improved linear interpolation support."""
    arg_parser = create_omop_meds_tutorial_arg_parser()
    
    arg_parser.add_argument(
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=str,
        default=None,
        help="Directory to resume training from checkpoint"
    )

    arg_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to save model outputs and checkpoints"
    )
    
    arg_parser.add_argument(
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for training"
    )
    
    arg_parser.add_argument(
        "--n_layers",
        dest="n_layers",
        type=int,
        default=11,
        help="Number of transformer layers"
    )
    
    arg_parser.add_argument(
        "--n_epochs",
        dest="n_epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    arg_parser.add_argument(
        "--per_device_train_batch_size",
        dest="per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device"
    )
    
    arg_parser.add_argument(
        "--per_device_eval_batch_size",
        dest="per_device_eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size per device"
    )
    
    # Enhanced linear interpolation argument with better type handling
    arg_parser.add_argument(
        "--linear_interpolation",
        dest="linear_interpolation",
        action="store_true",  # Use store_true for better boolean handling
        default=False,
        help="Enable linear interpolation for censoring adjustment in MOTOR task"
    )
    
    # Additional arguments for enhanced functionality
    arg_parser.add_argument(
        "--run_name",
        dest="run_name",
        type=str,
        default="motor_pretrain_v2",
        help="W&B run name for experiment tracking"
    )
    
    arg_parser.add_argument(
        "--early_stopping_patience",
        dest="early_stopping_patience",
        type=int,
        default=3,
        help="Early stopping patience in epochs"
    )
    
    arg_parser.add_argument(
        "--early_stopping_threshold",
        dest="early_stopping_threshold",
        type=float,
        default=0.001,
        help="Early stopping threshold for metric improvement"
    )
    
    return arg_parser


def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EnhancedWandbTrainLossCallback(transformers.TrainerCallback):
    """Enhanced W&B logging callback with better metrics tracking."""
    
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
        
        # Learning rate tracking
        if "learning_rate" in logs:
            log_data["train/learning_rate"] = logs["learning_rate"]
        
        # Record step and epoch for W&B axes
        log_data["global_step"] = state.global_step
        log_data["epoch"] = state.epoch
        
        # Log linear interpolation setting (static info)
        if hasattr(state, 'linear_interpolation'):
            log_data["config/linear_interpolation"] = state.linear_interpolation
        
        wandb.log(log_data)


def main():
    """Enhanced main function with improved linear interpolation handling."""
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)

    print(f"Starting MOTOR pretraining with linear interpolation: {args.linear_interpolation}")
    print(f"Output directory: {args.output_dir}")

    # Load ontology
    ontology_path = pretraining_data / 'ontology.pkl'
    with open(ontology_path, 'rb') as f:
        ontology = pickle.load(f)

    # Load tokenizer
    tokenizer_path = pretraining_data / 'tokenizer'
    tokenizer = femr.models.tokenizer.HierarchicalTokenizer.from_pretrained(
        tokenizer_path, ontology=ontology
    )
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load motor task
    task_path = pretraining_data / 'motor_task.pkl'
    with open(task_path, 'rb') as f:
        motor_task = pickle.load(f)
    print(f"Motor task: {motor_task}")
    print(f"Motor task length: {len(motor_task.pretraining_task_codes)}")
    
    # Create processor
    processor = femr.models.processor.FEMRBatchProcessor(tokenizer, motor_task)

    # Load datasets
    train_batches_path = pretraining_data / 'train_batches'
    train_batches = datasets.Dataset.load_from_disk(str(train_batches_path))
    print(f"Train batches length: {len(train_batches)}")

    val_batches_path = pretraining_data / 'val_batches'
    val_batches = datasets.Dataset.load_from_disk(str(val_batches_path))
    print(f"Val batches length: {len(val_batches)}")

    # Create enhanced transformer configuration
    transformer_config = femr.models.config.FEMRTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        is_hierarchical=isinstance(tokenizer, femr.models.tokenizer.HierarchicalTokenizer),
        n_layers=args.n_layers,
        use_normed_ages=True,
        use_bias=False,
        hidden_act='swiglu',
    )

    # Create model configuration
    config = femr.models.config.FEMRModelConfig.from_transformer_task_configs(
        transformer_config,
        motor_task.get_task_config()
    )

    print(f"Creating model with linear interpolation: {args.linear_interpolation}")
    
    # Create model using the enhanced FEMRModel with proper linear interpolation support
    model = femr_transformer.FEMRModel(
        linear_interpolation=args.linear_interpolation,
        config=config,
        attn_implementation="flash_attention_2"
    )
    model = model.to(torch.device("cuda:0"))

    print(f"Model parameter count: {count_parameters(model):,}")
    print(f"Linear interpolation enabled in model: {model.linear_interpolation}")

    # Enhanced training configuration
    output_dir = args.output_dir
    run_name_suffix = "_linear_interp" if args.linear_interpolation else "_no_linear_interp"
    full_run_name = f"{args.run_name}{run_name_suffix}"
    
    trainer_config = transformers.TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        output_dir=output_dir,
        remove_unused_columns=False,
        bf16=True,
        weight_decay=0.1,
        adam_beta2=0.95,
        report_to=["wandb"],
        run_name=full_run_name,
        num_train_epochs=args.n_epochs,
        ddp_find_unused_parameters=False,
        warmup_steps=500,
        logging_strategy='epoch',
        logging_steps=10,
        save_strategy='epoch',
        eval_strategy='epoch',
        dataloader_num_workers=64,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    print(f"Training configuration:")
    print(f"  Batch size per device: {trainer_config.per_device_train_batch_size}")
    print(f"  Learning rate: {trainer_config.learning_rate}")
    print(f"  Number of epochs: {trainer_config.num_train_epochs}")
    print(f"  Run name: {full_run_name}")

    # Enhanced callbacks with linear interpolation info
    callbacks = [
        CustomEarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience, 
            early_stopping_threshold=args.early_stopping_threshold
        ),
        EnhancedWandbTrainLossCallback()
    ]

    # Create trainer
    trainer = transformers.Trainer(
        model=model,
        data_collator=processor.collate,
        train_dataset=train_batches,
        eval_dataset=val_batches,
        args=trainer_config,
        callbacks=callbacks,
    )

    # Store linear interpolation setting in trainer state for logging
    trainer.state.linear_interpolation = args.linear_interpolation

    # Log initial configuration to W&B
    if wandb.run is not None:
        wandb.config.update({
            "linear_interpolation": args.linear_interpolation,
            "model_parameters": count_parameters(model),
            "vocab_size": tokenizer.vocab_size,
            "n_layers": args.n_layers,
            "motor_task_count": len(motor_task.pretraining_task_codes),
            "train_batches": len(train_batches),
            "val_batches": len(val_batches),
        })

    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    
    # Log and save results
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Save final model with linear interpolation info in the directory name
    final_model_dir = pathlib.Path(output_dir) / "final_model"
    if args.linear_interpolation:
        final_model_dir = pathlib.Path(output_dir) / "final_model_linear_interp"
    
    trainer.save_model(str(final_model_dir))
    print(f"Final model saved to: {final_model_dir}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()


# Enhanced usage examples with linear interpolation:
"""
Basic training without linear interpolation:
python pretrain_motor_v2.py \
  --pretraining_data /path/to/motor_data \
  --meds_reader /path/to/meds_reader \
  --per_device_train_batch_size 1 \
  --output_dir /path/to/output

Training with linear interpolation enabled:
python pretrain_motor_v2.py \
  --pretraining_data /path/to/motor_data \
  --meds_reader /path/to/meds_reader \
  --per_device_train_batch_size 1 \
  --output_dir /path/to/output \
  --linear_interpolation \
  --run_name "motor_linear_interp_experiment"

Multi-GPU training with linear interpolation:
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
  --num_processes 3 \
  --mixed_precision bf16 \
  --gpu_ids "0,1,2" \
  pretrain_motor_v2.py \
  --pretraining_data /path/to/motor_data \
  --meds_reader /path/to/meds_reader \
  --per_device_train_batch_size 1 \
  --output_dir /path/to/output \
  --linear_interpolation \
  --n_epochs 100 \
  --learning_rate 5e-6
"""
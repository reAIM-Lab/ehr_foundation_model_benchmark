# Linear probing for llama model trained in stanford
import argparse
from transformers import AutoModelForCausalLM
from hf_ehr.data.tokenization import CLMBRTokenizer
import torch
from pathlib import Path
import polars as pl

import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pyl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import LRModelLightning, load_meds, standardize

import warnings
warnings.filterwarnings("ignore")

####################################
# 1. Load model and tokenizer
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(f"StanfordShahLab/{args.model}").to(device)
    tokenizer = CLMBRTokenizer.from_pretrained(f"StanfordShahLab/{args.model}")

    base_path = Path(args.input_meds)
    task_name = args.task

    # load labels
    task_path = base_path / f"task_labels/cehrbert_pyspark/{task_name}/train"
    task_path_test = base_path / f"task_labels/cehrbert_pyspark/{task_name}/test"

    embedding_train_path = base_path / "context_clues_embeddings" / task_name / "train" 
    embedding_tune_path = base_path / "context_clues_embeddings" / task_name / "tune" 
    embedding_test_path = base_path / "context_clues_embeddings" / task_name / "test"  

    embedding_train_path.mkdir(parents=True, exist_ok=True)
    embedding_tune_path.mkdir(parents=True, exist_ok=True)
    embedding_test_path.mkdir(parents=True, exist_ok=True)

    train = pl.read_parquet(sorted(task_path.glob("*.parquet")), columns=["subject_id", "prediction_time", "boolean_value"])
    test = pl.read_parquet(sorted(task_path_test.glob("*.parquet")), columns=["subject_id", "prediction_time", "boolean_value"])

    print(len(train))
    print(len(test))

    train_path = base_path / "post_transform/data/train"
    tune_path = base_path / "post_transform/data/tuning"
    test_path = base_path / "post_transform/data/held_out"

    train_embeddings, train_labels = load_meds(train, train_path, model, tokenizer, embedding_train_path, device)
    tune_embeddings, tune_labels = load_meds(train, tune_path, model, tokenizer, embedding_tune_path, device)
    test_embeddings, test_labels = load_meds(test, test_path, model, tokenizer, embedding_test_path, device)

    train_embeddings, tune_embeddings, test_embeddings = standardize(train_embeddings, tune_embeddings, test_embeddings)

    batch_size = 256
    train_dataset = TensorDataset(train_embeddings, train_labels) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(tune_embeddings, tune_labels) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_embeddings, test_labels) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    input_dim = train_embeddings.shape[1]  # This should match the embedding size
    model = LRModelLightning(input_dim)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",       # Track validation loss
        mode="min",               # Save the model with the lowest val_loss
        save_top_k=1,             # Keep only the best model
        dirpath="checkpoints/",   # Save directory
        filename="best_model",    # Model file name
        verbose=True
    )

    trainer = pyl.Trainer(
        max_epochs=50, 
        accelerator='gpu', 
        devices=1, 
        logger=TensorBoardLogger("tb_logs"),
        callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_loader, val_loader)

    model = LRModelLightning.load_from_checkpoint(checkpoint_callback.best_model_path, input_dim=input_dim)
    trainer.test(model, test_loader)

    # Save the model
    torch.save(model.state_dict(), f"models/linear_probing_{args.task}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for Context Clues linear probing"
    )
    parser.add_argument(
        "--input_meds",
        dest="input_meds",
        action="store",
        default="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/"
    )

    parser.add_argument(
        "--task",
        dest="task",
        action="store",
        default="hospitalization_mortality_meds",
    )

    parser.add_argument(
        "--model",
        dest="model",
        action="store",
        default="llama-base-2048-clmbr",
    )

    main(
        parser.parse_args()
    )
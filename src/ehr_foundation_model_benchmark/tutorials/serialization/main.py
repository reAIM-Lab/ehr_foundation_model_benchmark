# Serialization based on https://arxiv.org/pdf/2502.17403
import argparse
from pathlib import Path
import polars as pl

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pyl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import standardize, normalize_icd10, load_embeddings, LRModelLightning

import warnings
warnings.filterwarnings("ignore")

def main(args):

    base_path = Path(args.input_meds)

    train_path = base_path / "post_transform/data/train"
    tune_path = base_path / "post_transform/data/tuning"
    test_path = base_path / "post_transform/data/held_out"

    # Load OMOP Concept table
    concept_path = Path("/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3/concept")
    concepts = pl.read_parquet(sorted(concept_path.glob("*.parquet")))
    concepts = concepts.with_columns(
        (pl.col("vocabulary_id").cast(pl.Utf8) + "/" + pl.col("concept_code").cast(pl.Utf8)).alias("vocabulary_concept")
    )
    concepts = concepts[['concept_name', 'vocabulary_concept']]
    concepts = normalize_icd10(concepts, 'vocabulary_concept')

    task_name = args.task

    # load labels
    task_path = base_path / f"task_labels/patient_outcomes_sample/{task_name}"
    task_path_test = base_path / f"task_labels/patient_outcomes_sample/{task_name}"

    train_subjects = pl.read_parquet(sorted(task_path.glob("*.parquet")))
    test_subjects = pl.read_parquet(sorted(task_path_test.glob("*.parquet")))

    train_files = sorted(train_path.glob("*.parquet"))
    tune_files = sorted(tune_path.glob("*.parquet"))
    test_files = sorted(test_path.glob("*.parquet"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-1.5B-instruct', trust_remote_code=True)
    model = model.to(device)

    embedding_train_path = base_path / "serialization_embeddings" / task_name / "train" 
    embedding_tune_path = base_path / "serialization_embeddings" / task_name / "tune" 
    embedding_test_path = base_path / "serialization_embeddings" / task_name / "test"  

    embedding_train_path.mkdir(parents=True, exist_ok=True)
    embedding_tune_path.mkdir(parents=True, exist_ok=True)
    embedding_test_path.mkdir(parents=True, exist_ok=True)

    task = "Given a patient's electronic healthcare record in Markdown format, retrieve relevant passages that answer the query"

    # Load and filter each parquet file in data_path
    train_embeddings, train_labels = load_embeddings(train_subjects, concepts, train_files, embedding_train_path, model, tokenizer, task, device, args)
    tune_embeddings, tune_labels = load_embeddings(train_subjects, concepts, tune_files, embedding_tune_path, model, tokenizer, task, device, args)
    test_embeddings, test_labels = load_embeddings(test_subjects, concepts, test_files, embedding_test_path, model, tokenizer, task, device, args)

    train_embeddings, tune_embeddings, test_embeddings = standardize(train_embeddings, tune_embeddings, test_embeddings)

    print(train_embeddings.shape)
    print(test_embeddings.shape)

    batch_size = 256
    train_dataset = TensorDataset(train_embeddings, train_labels.long()) 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(tune_embeddings, tune_labels.long()) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_embeddings, test_labels.long()) 
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

    model_path = Path(f"models/linear_probing_{args.task}.pth")

    if not model_path.exists():
        trainer = pyl.Trainer(
            max_epochs=50, 
            accelerator='gpu', 
            devices=1, 
            logger=TensorBoardLogger("tb_logs"),
            callbacks=[checkpoint_callback])
        
        trainer.fit(model, train_loader, val_loader)

        model = LRModelLightning.load_from_checkpoint(checkpoint_callback.best_model_path, input_dim=input_dim)
        
        # Save the model
        torch.save(model.state_dict(), model_path)

    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

    trainer = pyl.Trainer(accelerator='gpu', devices=1)  
    trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for Text Serialization"
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
        default="readmission",
    )

    main(
        parser.parse_args()
    )
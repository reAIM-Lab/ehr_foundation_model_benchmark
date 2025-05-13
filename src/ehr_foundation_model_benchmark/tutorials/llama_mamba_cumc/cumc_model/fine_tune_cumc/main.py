import argparse
import time
import warnings
from pathlib import Path
import pandas as pd
import polars as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pyl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import LRModelLightning, load_task_embeddings

warnings.filterwarnings("ignore")

TRAIN_SIZES = [1000, 10000, 100000]

####################################
# 1. Load model and tokenizer
def main(args):
    device = args.device
    # torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)

    base_path = Path(args.input_meds)
    task_name = args.task

    # Load in outcome cohorts with labels and prediction time
    if task_name in ["AMI", "Celiac", "CLL", "HTN", "Ischemic_Stroke", "MASLD", "Osteoporosis", "Pancreatic_Cancer", "SLE", "T2DM"]:
        task_path_train = base_path / f"task_labels/phenotype_sample/{task_name}" / "train.parquet"
        task_path_val = base_path / f"task_labels/phenotype_sample/{task_name}" / "tuning.parquet"
        task_path_test = base_path / f"task_labels/phenotype_sample/{task_name}" / "held_out.parquet"

        train = pl.read_parquet(task_path_train, columns=["subject_id", "prediction_time", "boolean_value"])
        tune = pl.read_parquet(task_path_val, columns=["subject_id", "prediction_time", "boolean_value"])
        test = pl.read_parquet(task_path_test)
    else:
        task_path_train = base_path / f"task_labels/patient_outcomes_sample/{task_name}" / "train.parquet"
        task_path_val = base_path / f"task_labels/patient_outcomes_sample/{task_name}" / "tuning.parquet"
        task_path_test = base_path / f"task_labels/patient_outcomes_sample/{task_name}" / "held_out.parquet"

        train = pl.read_parquet(task_path_train, columns=["subject_id", "prediction_time", "boolean_value"])
        tune = pl.read_parquet(task_path_val, columns=["subject_id", "prediction_time", "boolean_value"])
        test = pl.read_parquet(task_path_test)

    train_path = base_path / "post_transform/data/train"
    tune_path = base_path / "post_transform/data/tuning"
    test_path = base_path / "post_transform/data/held_out"

    print(len(train))
    print(len(tune))
    print(len(test))

    # Load in embeddings and labels for task (model dependent)
    start_time = time.time()

    train_embeddings, train_labels, _, _ = load_task_embeddings(args, train, train_path, device,"train")
    tune_embeddings, tune_labels, _, _ = load_task_embeddings(args, tune, tune_path, device,"tune")
    test_embeddings, test_labels, test_ids, test_times = load_task_embeddings(args, test, test_path, device,"test")
    
    print(len(train_labels))
    print(len(tune_labels))
    print(len(test_labels))

    # combine train and tune embeddings and labels
    base_path = Path(args.input_meds)
    output_dir = base_path / args.model_type / args.task/ "features_with_label" 

    train_path   = output_dir  / "train.parquet"
    tune_path    = output_dir  / "tune.parquet"

    # 1) load
    df_train = pd.read_parquet(train_path)
    df_tune  = pd.read_parquet(tune_path)

    # 2) concat & sort
    df = pd.concat([df_train, df_tune], ignore_index=True)
    df = df.sort_values(by=["subject_id", "prediction_time"])

    # 3) overwrite train.parquet
    df.to_parquet(train_path, index=False)


    end_time = time.time()
    print(f"Inference Time taken: {end_time - start_time:.2f} seconds")

    # train_embeddings, tune_embeddings, test_embeddings = standardize(train_embeddings, tune_embeddings, test_embeddings)

    # model_dir = Path(f"./models/{task_name}")
    # model_dir.mkdir(parents=True, exist_ok=True)

    # pred_dir = Path(f"./predictions/{task_name}")
    # pred_dir.mkdir(parents=True, exist_ok=True)

    # for size in TRAIN_SIZES:
    #     indices = torch.randperm(train_embeddings.shape[0])[:size]

    #     x_train = train_embeddings[indices]
    #     y_train = train_labels[indices]

    #     batch_size = 256
    #     train_dataset = TensorDataset(x_train, y_train) 
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     val_dataset = TensorDataset(tune_embeddings, tune_labels) 
    #     val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    #     test_dataset = TensorDataset(test_embeddings, test_labels) 
    #     test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    #     unique_labels, counts = test_labels.unique(return_counts=True)
    #     label_prevalence = counts.float() / len(test_labels)

    #     # Display the label prevalences
    #     for label, prevalence in zip(unique_labels, label_prevalence):
    #         print(f"Label '{label.item()}' prevalence: {prevalence:.4f}")

    #     # Train the model
    #     input_dim = train_embeddings.shape[1]  # This should match the embedding size
    #     model = LRModelLightning(input_dim)

    #     checkpoint_callback = ModelCheckpoint(
    #         monitor="val_loss",       # Track validation loss
    #         mode="min",               # Save the model with the lowest val_loss
    #         save_top_k=1,             # Keep only the best model
    #         dirpath="checkpoints/",   # Save directory
    #         filename="best_model",    # Model file name
    #         verbose=False
    #     )

    #     early_stopping_callback = EarlyStopping(
    #         monitor="val_loss",    # Metric to monitor (validation loss)
    #         patience=20,            # Number of epochs to wait for improvement
    #         verbose=False,          # Print message when stopping
    #         mode="min",            # 'min' for loss (lower is better)
    #     )

    #     model_path = model_dir / f"linear_probing_{size}.pth"

    #     if not model_path.exists():
    #         trainer = pyl.Trainer(
    #             max_epochs=100, 
    #             accelerator='gpu', 
    #             devices=1, 
    #             callbacks=[checkpoint_callback, early_stopping_callback])
            
    #         trainer.fit(model, train_loader, val_loader)

    #         model = LRModelLightning.load_from_checkpoint(checkpoint_callback.best_model_path, input_dim=input_dim)
            
    #         # Save the model
    #         torch.save(model.state_dict(), model_path)

    #     model.load_state_dict(torch.load(model_path))
    #     print(f"Loaded model from {model_path}")

    #     trainer = pyl.Trainer(accelerator='gpu', devices=1)  
    #     trainer.test(model, test_loader)

    #     results = trainer.predict(model, test_loader)
    #     preds = results[0][0]
    #     probs = results[0][1]

    #     df_predictions = pl.DataFrame({
    #         "subject_id": np.array(test_ids).astype(int),
    #         "prediction_time": test_times,
    #         "boolean_value": test_labels.cpu().numpy().astype(bool),
    #         "predicted_boolean_value": preds.detach().cpu().numpy().astype(bool),
    #         "predicted_boolean_probability": probs.detach().cpu().numpy().astype(float),
    #     })

    #     prediction_save_path = pred_dir / f"{args.model_type}_{size}.parquet"
    #     df_predictions.write_parquet(prediction_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for Context Clues linear probing"
    )
    parser.add_argument(
        "--input_meds",
        dest="input_meds",
        action="store",
        default="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped",
        help="Path to input MEDS data"
    )

    parser.add_argument(
        "--task",
        dest="task",
        action="store",
        default="AMI",
        help="Name of task file"
    )

    parser.add_argument(
        "--model",
        dest="model",
        action="store",
        default="mamba-tiny-8192-clmbr",
        help="Model name used for downloading off huggingface"
    )

    parser.add_argument(
        "--model_path",
        dest="model_path",
        action="store",
        default="",
        help="Path to pretrained model"
    )


    parser.add_argument(
        "--model_type",
        dest="model_type",
        action="store",
        default="mamba-ehrshot",
        help="Model name used for saving embeddings and predictions"
    )


    parser.add_argument(
        "--device",
        dest="device",
        action="store",
        default="cpu",
        help="Model name used for saving embeddings and predictions"
    )

    parser.add_argument(
        "--seed",
        dest="seed",
        action="store",
        default=123,
    )

    main(
        parser.parse_args()
    )
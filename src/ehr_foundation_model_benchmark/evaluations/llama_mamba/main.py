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
    tune_path.unlink()


    end_time = time.time()
    print(f"Inference Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for Context Clues linear probing"
    )
    parser.add_argument(
        "--input_meds",
        dest="input_meds",
        action="store",
        default=None,
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
        default="mamba",
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
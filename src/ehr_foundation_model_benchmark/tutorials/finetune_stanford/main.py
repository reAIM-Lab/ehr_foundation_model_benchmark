import argparse
import time
import warnings
from pathlib import Path

import polars as pl
import numpy as np
import torch

from utils import load_task_embeddings

warnings.filterwarnings("ignore")

TRAIN_SIZES = [1000, 10000, 100000]

####################################
# Generating embeddings from pretrained model
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Load in embeddings and labels for task (model dependent)
    start_time = time.time()

    train_embeddings, train_labels, train_ids, train_times = load_task_embeddings(args, train, train_path, device)
    tune_embeddings, tune_labels, tune_ids, tune_times = load_task_embeddings(args, tune, tune_path, device)
    test_embeddings, test_labels, test_ids, test_times = load_task_embeddings(args, test, test_path, device)

    end_time = time.time()
    print(f"Inference Time taken: {end_time - start_time:.2f} seconds")

    train_features = pl.DataFrame({
            "subject_id": np.array(train_ids).astype(int),
            "prediction_time": train_times,
            "boolean_value": train_labels.cpu().numpy().astype(bool),
            "features": np.array(train_embeddings.detach().cpu())
        })

    tune_features = pl.DataFrame({
            "subject_id": np.array(tune_ids).astype(int),
            "prediction_time": tune_times,
            "boolean_value": tune_labels.cpu().numpy().astype(bool),
            "features": np.array(tune_embeddings.detach().cpu())
        })

    test_features = pl.DataFrame({
            "subject_id": np.array(test_ids).astype(int),
            "prediction_time": test_times,
            "boolean_value": test_labels.cpu().numpy().astype(bool),
            "features": np.array(test_embeddings.detach().cpu())
        })
    all_features = pl.concat([train_features, tune_features, test_features])

    output_path = Path(f"outputs/{args.task}/features_with_label")
    output_path.mkdir(parents=True, exist_ok=True)
    all_features.write_parquet(output_path / f"{args.task}.parquet")

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
        "--model_type",
        dest="model_type",
        action="store",
        default="mamba-ehrshot",
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
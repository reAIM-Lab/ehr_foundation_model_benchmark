"""
Variant from linear_prob.py to read meds-tab features
The subsampling strategy is slightly different to ensure a minimum of cases in train and tuning, instead of doing cross validation
Only compute the subsampled splits, because meds-tab will perform the XGB training
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import json
import pickle
from meds import train_split, tuning_split, held_out_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

import scipy.sparse as sp

MINIMUM_NUM_CASES_TRAIN = 8
MINIMUM_NUM_CASES_TUNING = 2
TRAIN_SIZES = [80, 800, 8000]
TUNING_SIZES = [20, 200, 2000]

# TODO import from medstab
def load_tab(path):
    """Loads a sparse matrix from disk.

    Args:
        path: Path to the sparse matrix.

    Returns:
        The sparse matrix.

    Raises:
        ValueError: If the loaded array does not have exactly 3 rows, indicating an unexpected format.
    """
    npzfile = np.load(path)
    array, shape = npzfile["array"], npzfile["shape"]
    if array.shape[0] != 3:
        raise ValueError(f"Expected array to have 3 rows, but got {array.shape[0]} rows")
    data, row, col = array
    return sp.csc_matrix((data, (row, col)), shape=shape)

def main(args):
    meds_dir = Path(args.meds_dir)
    subject_splits_path = meds_dir / "metadata" / "subject_splits.parquet"
    print(f"Loading subject_splits.parquet from {subject_splits_path}")
    subject_splits = pl.read_parquet(subject_splits_path)
    features_label_input_dir = Path(args.features_label_input_dir)
    # features_label = pl.read_parquet(list(features_label_input_dir.rglob('*.parquet')))
    features_label = pl.read_parquet(Path(args.features_label_input_dir) / "indices.parquet")
    # add index columns
    features_label = features_label.with_row_index(
        name="sample_id",
        offset=0
    )
    print(f"Loading features_label from {features_label_input_dir}")

    output_dir = Path(args.output_dir)
    task_output_dir = output_dir / args.task_name
    task_output_dir.mkdir(exist_ok=True, parents=True)

    features_label = features_label.sort("subject_id", "prediction_time")

    # print(features_label)
    # print(subject_splits)

    train_dataset = features_label.join(
        subject_splits.select("subject_id", "split"), "subject_id"
    ).filter(
        pl.col("split").is_in([train_split])
    )
    original_positive_prevalence_train = train_dataset.filter(
        pl.col("boolean_value") == True
    ).shape[0] / train_dataset.shape[0]
    print(f"Original positive prevalence: {original_positive_prevalence_train}")

    tuning_dataset = features_label.join(
        subject_splits.select("subject_id", "split"), "subject_id"
    ).filter(
        pl.col("split").is_in([tuning_split])
    )
    original_positive_prevalence_tuning = tuning_dataset.filter(
        pl.col("boolean_value") == True
    ).shape[0] / tuning_dataset.shape[0]
    print(f"Original positive prevalence tuning: {original_positive_prevalence_tuning}")

    test_dataset = features_label.join(
        subject_splits.select("subject_id", "split"), "subject_id"
    ).filter(
        pl.col("split") == held_out_split
    )

    # feature_matrix = load_tab(
    #     Path(args.features_label_input_dir) / "features_combined.npz"
    # )
    # print(feature_matrix.shape)

    should_terminate = False
    # We keep track of the sample ids that have been picked from the previous few-shots experiments.
    existing_sample_ids = set()
    existing_sample_ids_tuning = set()
    for i, size in enumerate(TRAIN_SIZES):
        print("_" * 20)
        # This indicates the data set has reached its maximum size, and we should terminate
        if should_terminate:
            break

        if len(train_dataset) < size:
            size = len(train_dataset)
            should_terminate = True

        test_prediction_parquet_file = task_output_dir / f"{args.model_name}_{size}.parquet"
        few_show_output_dir = task_output_dir / f"{args.model_name}_{size}"
        few_show_output_dir.mkdir(exist_ok=True, parents=True)
        logistic_model_file = few_show_output_dir / "model.pickle"
        logistic_test_metrics_file = few_show_output_dir / "metrics.json"

        if logistic_test_metrics_file.exists():
            print(
                f"The results for logistic regression with {size} shots already exist at {logistic_test_metrics_file}"
            )
        else:
            remaining_train_set = train_dataset.filter(~pl.col("sample_id").is_in(existing_sample_ids))
            remaining_tuning_set = tuning_dataset.filter(~pl.col("sample_id").is_in(existing_sample_ids_tuning))

            existing_samples = train_dataset.filter(pl.col("sample_id").is_in(existing_sample_ids))
            existing_samples_tuning = tuning_dataset.filter(pl.col("sample_id").is_in(existing_sample_ids_tuning))
            try:
                existing_pos = len(existing_samples.filter(pl.col("boolean_value") == True))
                train_size_required_positive = \
                    int(original_positive_prevalence_train * size) - existing_pos
                if train_size_required_positive + existing_pos < MINIMUM_NUM_CASES_TRAIN:
                    if len(remaining_train_set.filter(pl.col("boolean_value") == True)) > MINIMUM_NUM_CASES_TRAIN:
                        train_size_required_positive = max(MINIMUM_NUM_CASES_TRAIN - existing_pos, 0)
                    else:
                        print(
                            f"The number of positive cases is less than {MINIMUM_NUM_CASES_TRAIN} for {size}"
                        )
                        continue
                train_size_required_negative = \
                    size - train_size_required_positive - len(existing_samples.filter(pl.col("boolean_value") == False)) - existing_pos

                subset = pl.concat([
                    remaining_train_set.filter(pl.col("boolean_value") == True).sample(
                        n=train_size_required_positive, shuffle=True, seed=args.seed
                    ),
                    remaining_train_set.filter(pl.col("boolean_value") == False).sample(
                        n=train_size_required_negative, shuffle=True, seed=args.seed
                    ),
                    existing_samples
                ]).sample(
                    fraction=1.0,
                    shuffle=True,
                    seed=args.seed
                )

                existing_pos_tuning = len(existing_samples_tuning.filter(pl.col("boolean_value") == True))
                tuning_size_required_positive = \
                    int(original_positive_prevalence_tuning * TUNING_SIZES[i]) - existing_pos_tuning
                if tuning_size_required_positive + existing_pos_tuning < MINIMUM_NUM_CASES_TUNING:
                    if len(remaining_tuning_set.filter(pl.col("boolean_value") == True)) > MINIMUM_NUM_CASES_TUNING:
                        tuning_size_required_positive = max(MINIMUM_NUM_CASES_TUNING - existing_pos_tuning, 0)
                    else:
                        print(
                            f"The number of positive cases is less than {MINIMUM_NUM_CASES_TUNING} for {size}"
                        )
                        continue
                tuning_size_required_negative = \
                    TUNING_SIZES[i] - tuning_size_required_positive - len(existing_samples_tuning.filter(pl.col("boolean_value") == False)) - existing_pos_tuning
                subset_tuning = pl.concat([
                    remaining_tuning_set.filter(pl.col("boolean_value") == True).sample(
                        n=tuning_size_required_positive, shuffle=True, seed=args.seed
                    ),
                    remaining_tuning_set.filter(pl.col("boolean_value") == False).sample(
                        n=tuning_size_required_negative, shuffle=True, seed=args.seed
                    ),
                    existing_samples_tuning
                ]).sample(
                    fraction=1.0,
                    shuffle=True,
                    seed=args.seed
                )               

                existing_sample_ids.update(subset["sample_id"].to_list())
                existing_sample_ids_tuning.update(subset_tuning["sample_id"].to_list())
                # count per class for train and tuning
                count_by_class = subset.group_by("boolean_value").count().to_dict(as_series=False)
                count_by_class_tuning = subset_tuning.group_by("boolean_value").count().to_dict(as_series=False)
                print(f"Train set: {count_by_class}")
                print(f"Tuning set: {count_by_class_tuning}")

                print(subset)
                print(subset_tuning)
                
                subset_all = pl.concat([
                    subset,
                    subset_tuning
                ])

                print(subset_all)

                subset_all.write_parquet(task_output_dir / f"{args.model_name}_{size+TUNING_SIZES[i]}.parquet")
                print(f"Saved to ", task_output_dir / f"{args.model_name}_{size+TUNING_SIZES[i]}.parquet")

                
            except ValueError as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for Context Clues linear probing"
    )
    parser.add_argument(
        "--features_label_input_dir",
        dest="features_label_input_dir",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--meds_dir",
        dest="meds_dir",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        action="store",
        default=42,
    )
    parser.add_argument(
        "--model_name",
        dest="model_name",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--task_name",
        dest="task_name",
        action="store",
        required=True,
    )
    main(
        parser.parse_args()
    )
"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""
import json
import meds_reader
from pathlib import Path
import pandas as pd
import polars as pl
import femr.featurizers
import pickle
import pathlib
import os
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
import femr.splits
import argparse
from .generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser
from .generate_motor_features import get_model_name

from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv
import torchtuples as tt

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)


def create_arg_parser():
    # args = create_omop_meds_tutorial_arg_parser()
    args = argparse.ArgumentParser(description="Arguments for linear probing")
    args.add_argument(
        "--meds_reader",
        dest="meds_reader",
        default=None,
    )
    args.add_argument(
        "--cohort_label",
        dest="cohort_label",
        default=None,
    )
    args.add_argument(
        "--observation_window",
        dest="observation_window",
        type=int,
        default=None,
        help="The observation window for extracting features",
    )

    args.add_argument(
        "--main_split_path",
        dest="main_split_path", 
        default=None, 
        help="The path to the main split file",
    )

    args.add_argument(
        "--model_name",
        dest="model_name",
        required=True,
        help="The model name",
    )
    args.add_argument(
        "--model_path",
        dest="model_path",
        required=True,
        help="The model path"
    )
    args.add_argument(
        "--output_root",
        default=None,
        help="Directory where results/features_with_label will be written. Defaults to --pretraining_data.",
    )

    return args

def logistic_regression(train_set,test_data,label_output_dir):
    model = LogisticRegressionCV(scoring='roc_auc')
    model.fit(train_set['features'].to_list(), train_set['boolean_value'])

    y_pred = model.predict_proba(test_data['features'])[:, 1]

    # Convert predictions to a Polars DataFrame
    logistic_predictions = pl.DataFrame({
        "subject_id": test_data["subject_ids"].tolist(),
        "prediction_time": test_data["prediction_times"].tolist(),
        "predicted_boolean_probability": y_pred.tolist(),
        "predicted_boolean_value": None,
        "boolean_value": test_data["boolean_values"].astype(bool).tolist()
    })

    logistic_predictions = logistic_predictions.with_columns(
        pl.col("predicted_boolean_value").cast(pl.Boolean())
    )

    # Create output directory
    logistic_test_predictions = label_output_dir / "test_predictions"
    logistic_test_predictions.mkdir(exist_ok=True, parents=True)
    # Write to parquet
    logistic_predictions.write_parquet(logistic_test_predictions / "predictions.parquet")

    roc_auc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], y_pred)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(test_data['boolean_values'], y_pred)
    pr_auc = sklearn.metrics.auc(recall, precision)
    return roc_auc,pr_auc

    
def main():
    args = create_arg_parser().parse_args()
    # pretraining_data = pathlib.Path(args.pretraining_data)
    label_names = LABEL_NAMES
    output_root = Path(args.output_root)
    if args.cohort_label is not None:
        label_path = output_root / "labels" / (args.cohort_label + '.parquet')
        if label_path.exists():
            print(f"Using the user defined label at: {label_path}")
            label_names = [args.cohort_label]
        else:
            raise RuntimeError(f"The user provided label does not exist at {label_path}")

    # labels = ["in_hospital_mortality","readmission","long_los"]
    output_dir = Path(output_root) / "results"
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=32) as database:
        for label_name in label_names:
            if args.observation_window:
                label_output_dir = output_dir / label_name/ f"{args.model_name}_{args.observation_window}"
            else:
                label_output_dir = output_dir / label_name/ f"{args.model_name}"
            label_output_dir.mkdir(exist_ok=True, parents=True)
            test_result_file = label_output_dir / 'metrics.json'
            features_label_data = label_output_dir / 'features_with_label'
            features_label_data.mkdir(exist_ok=True, parents=True)
            # if test_result_file.exists():
            #     print(f"The result already existed for {label_name} at {test_result_file}, it will be skipped!")
            #     continue
            labels = pd.read_parquet(output_root / "labels" / (label_name + '.parquet'))

            # motor_features_name = get_model_name(label_name, args.model_name, args.observation_window)
            features_path = output_root / "features"/ args.model_name
            with open(features_path / f"{label_name}_tpp.pkl", 'rb') as f:
                features = pickle.load(f)

            # Find labels that have no features
            labels_no_features = labels[~labels.subject_id.isin(features["subject_ids"])]
            if len(labels_no_features) > 0:
                print(f"{len(labels_no_features)} features are not included")
                # labels_no_features.to_parquet(label_output_dir / "labels_no_features.parquet")

            # Remove the labels that do not have features generated
            labels = labels[labels.subject_id.isin(features["subject_ids"])]
            labels = labels.sort_values(["subject_id", "prediction_time"])
            # labels = labels.sample(n=len(labels), random_state=42, replace=False)
            labeled_features = femr.featurizers.join_labels(features, labels,label_name)

            main_split = femr.splits.SubjectSplit.load_from_csv(args.main_split_path)

            # print(main_split)
            train_mask = np.isin(labeled_features['subject_ids'], main_split.train_subject_ids)
            test_mask = np.isin(labeled_features['subject_ids'], main_split.test_subject_ids)

            def apply_mask(values, mask):
                def apply(k, v):
                    if len(v.shape) == 1:
                        return v[mask]
                    elif len(v.shape) == 2:
                        return v[mask, :]
                    else:
                        assert False, f"Cannot handle {k} {v.shape}"

                return {k: apply(k, v) for k, v in values.items()}

            train_data = apply_mask(labeled_features, train_mask)
            test_data = apply_mask(labeled_features, test_mask)

            print("Saving features and labels to parquet")
            train_features_list = [feature for feature in train_data["features"]]
            test_features_list = [feature for feature in test_data["features"]]

            if label_name in ["ami","masld","stroke"]:
                train_set = pd.DataFrame({
                    "subject_id" : train_data["subject_ids"],
                    "prediction_time" : train_data["prediction_times"],
                    "tte_label" : train_data["tte_label"],
                    "time_to_event": train_data["time_to_event"],
                    "boolean_value": train_data["boolean_values"],
                    "features" : train_features_list
                }).sample(
                    frac=1.0,
                    random_state=42
                )
                train_set.to_parquet(features_label_data / "train.parquet", index=False)
                test_set = pd.DataFrame({
                    "subject_id" : test_data["subject_ids"],
                    "prediction_time" : test_data["prediction_times"],
                    "tte_label" : test_data["tte_label"],
                    "time_to_event": test_data["time_to_event"],
                    "boolean_value": test_data["boolean_values"],
                    "features" : test_features_list
                })
                test_set.to_parquet(features_label_data / "test.parquet", index=False)

            else:
                train_set = pd.DataFrame({
                    "subject_id" : train_data["subject_ids"],
                    "prediction_time" : train_data["prediction_times"],
                    "boolean_value": train_data["boolean_values"],
                    "features" : train_features_list
                }).sample(
                    frac=1.0,
                    random_state=42
                )
                train_set.to_parquet(features_label_data / "train.parquet", index=False)

                test_set = pd.DataFrame({
                    "subject_id" : test_data["subject_ids"],
                    "prediction_time" : test_data["prediction_times"],
                    "boolean_value": test_data["boolean_values"],
                    "features" : test_features_list
                })
                test_set.to_parquet(features_label_data / "test.parquet", index=False)


            print(f"Total labels: {len(labels)}")
            print(f"Total train features labels: {len(train_data['features'])}")
            print(f"Total test features labels: {len(test_data['features'])}")

            # model = LogisticRegressionCV(scoring='roc_auc')
            # model.fit(train_set['features'].to_list(), train_set['boolean_value'])

            # y_pred = model.predict_proba(test_data['features'])[:, 1]

            # # Convert predictions to a Polars DataFrame
            # logistic_predictions = pl.DataFrame({
            #     "subject_id": test_data["subject_ids"].tolist(),
            #     "prediction_time": test_data["prediction_times"].tolist(),
            #     "predicted_boolean_probability": y_pred.tolist(),
            #     "predicted_boolean_value": None,
            #     "boolean_value": test_data["boolean_values"].astype(bool).tolist()
            # })

            # logistic_predictions = logistic_predictions.with_columns(
            #     pl.col("predicted_boolean_value").cast(pl.Boolean())
            # )

            # # Create output directory
            # logistic_test_predictions = label_output_dir / "test_predictions"
            # logistic_test_predictions.mkdir(exist_ok=True, parents=True)
            # # Write to parquet
            # logistic_predictions.write_parquet(logistic_test_predictions / "predictions.parquet")

            # roc_auc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], y_pred)
            # precision, recall, _ = sklearn.metrics.precision_recall_curve(test_data['boolean_values'], y_pred)
            # pr_auc = sklearn.metrics.auc(recall, precision)

            roc_auc,pr_auc = logistic_regression(train_set,test_data,label_output_dir)
            print(label_name, roc_auc)
            metrics = {
                "auroc": roc_auc,
                "aucpr": pr_auc
            }
            with open(test_result_file, "w") as f:
                json.dump(metrics, f, indent=4)




if __name__ == "__main__":
    main()


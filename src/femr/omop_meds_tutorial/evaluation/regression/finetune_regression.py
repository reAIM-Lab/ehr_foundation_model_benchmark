"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""
import json
import meds_reader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import femr.splits
import argparse
from femr.omop_meds_tutorial.evaluation.generate_motor_features import get_model_name

from femr.omop_meds_tutorial.evaluation.generate_labels import (  # type: ignore
        create_omop_meds_tutorial_arg_parser,
        LABEL_NAMES,
    )

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


def main():
    args = create_arg_parser().parse_args()

    # Avoid variable shadowing: keep label_names for task iteration
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
                label_output_dir = output_dir / label_name / f"{args.model_name}_{args.observation_window}"
            else:
                label_output_dir = output_dir / label_name / f"{args.model_name}"
            label_output_dir.mkdir(exist_ok=True, parents=True)
            test_result_file = label_output_dir / 'metrics.json'
            features_label_data = label_output_dir / 'features_with_label'
            features_label_data.mkdir(exist_ok=True, parents=True)
            if test_result_file.exists():
                print(f"The result already existed for {label_name} at {test_result_file}, it will be skipped!")
                continue
            labels = pd.read_parquet(output_root / "labels" / (label_name + '.parquet'))

            motor_features_name = get_model_name(label_name, args.model_name, args.observation_window)
            features_path = output_root / "features"
            with open(features_path / f"{motor_features_name}.pkl", 'rb') as f:
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
            labeled_features = femr.featurizers.join_labels_numerical(features, labels)

            main_split = femr.splits.SubjectSplit.load_from_csv(args.main_split_path)

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
            train_set = (
                pd.DataFrame(
                    {
                        "subject_id": train_data["subject_ids"],
                        "prediction_time": train_data["prediction_times"],
                        "numerical_value": train_data["numerical_value"],
                        "features": train_features_list,
                    }
                )
                .sample(frac=1.0, random_state=42)
            )
            train_set.to_parquet(features_label_data / "train.parquet", index=False)
            test_features_list = [feature for feature in test_data["features"]]
            pd.DataFrame(
                {
                    "subject_id": test_data["subject_ids"],
                    "prediction_time": test_data["prediction_times"],
                    "numerical_value": test_data["numerical_value"],
                    "features": test_features_list,
                }
            ).to_parquet(features_label_data / "test.parquet", index=False)

            print(f"Total labels: {len(labels)}")
            print(f"Total train features labels: {len(train_data['features'])}")
            print(f"Total test features labels: {len(test_data['features'])}")

            # Prepare dense float32 arrays explicitly to avoid dtype/object issues
            X_train = train_data["features"]
            y_train = train_data["numerical_value"]
            X_test = test_data["features"]
            # if isinstance(X_train, list):
            #     X_train = np.asarray(X_train)
            # if isinstance(X_test, list):
            #     X_test = np.asarray(X_test)
            # X_train = X_train.astype(np.float32, copy=False)
            # X_test = X_test.astype(np.float32, copy=False)

            # print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
            ])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # model.fit(train_set['features'].to_list(), train_set["numerical_value"])
            # y_pred = model.predict(test_data['features'])

            pl.DataFrame({
                "subject_id": test_data["subject_ids"].tolist(),
                "prediction_time": test_data["prediction_times"].tolist(),
                "y_true": test_data["numerical_value"].tolist(),
                "y_pred": y_pred.tolist(),
                "residual": (test_data["numerical_value"] - y_pred).tolist(),
            }).write_parquet(label_output_dir / "test_predictions.parquet")
            rmse = float(np.sqrt(mean_squared_error(test_data["numerical_value"], y_pred)))
            mae = float(mean_absolute_error(test_data["numerical_value"], y_pred))
            r2 = float(r2_score(test_data["numerical_value"], y_pred))
            metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "alpha": float(model.named_steps["ridge"].alpha_),
            }
            print(f"{label_name} ridge α={metrics['alpha']:.4g} | RMSE={rmse:.4g} | MAE={mae:.4g} | R2={r2:.4g}")
            with open(test_result_file, "w") as f:
                json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()

# femr/omop_meds_tutorial/evaluation/finetune_motor_regression.py
import os
import sys
import json
import pathlib
import pickle

import argparse
import numpy as np
import pandas as pd
import polars as pl

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import meds_reader
import femr.splits

try:
    from .generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser
    from .generate_features_regression import get_motor_features_name
except Exception:
    _here = pathlib.Path(__file__).resolve()
    _repo_root = _here.parents[3]  # path that CONTAINS 'femr'
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from femr.omop_meds_tutorial.evaluation.generate_labels import (  # type: ignore
        LABEL_NAMES,
        create_omop_meds_tutorial_arg_parser,
    )
    from femr.omop_meds_tutorial.evaluation.regression.generate_features_regression import (  # type: ignore
        get_motor_features_name,
    )


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument("--cohort_label", dest="cohort_label", default=None)
    args.add_argument("--observation_window", dest="observation_window", type=int, default=None)
    args.add_argument("--main_split_path", dest="main_split_path", default=None)
    args.add_argument(
        "--output_root",
        default=None,
        help="Directory where results/features_with_label will be written. Defaults to --pretraining_data.",
    )
    # tolerate extra flags forwarded by the shell; they are no-ops here
    args.add_argument("--ontology_path", default=None, help=argparse.SUPPRESS)
    args.add_argument("--model_path", default=None, help=argparse.SUPPRESS)
    args.add_argument("--num_proc", type=int, default=None, help=argparse.SUPPRESS)
    args.add_argument("--tokens_per_batch", type=int, default=None, help=argparse.SUPPRESS)
    args.add_argument("--device", default=None, help=argparse.SUPPRESS)
    args.add_argument("--min_subjects_per_batch", type=int, default=None, help=argparse.SUPPRESS)
    args.add_argument("--linear_interpolation", dest="use_linear_interpolation",
                      action="store_true", help=argparse.SUPPRESS)
    return args


def temporal_same_day_strict_future_join(features: dict, labels: pd.DataFrame) -> dict:
    """
    features: dict with keys ['subject_ids', 'prediction_times', 'features']
    labels: DataFrame with ['subject_id','prediction_time','target_value']
    """
    if "prediction_times" not in features:
        raise RuntimeError("Features missing prediction_times; regenerate features with stamping enabled.")
    feat_times = pd.to_datetime(features["prediction_times"])

    fdf = pd.DataFrame({
        "subject_id": features["subject_ids"],
        "feature_time": feat_times,
        "feat_idx": np.arange(len(features["subject_ids"])),
    })
    fdf["date"] = fdf["feature_time"].dt.date

    ldf = labels.copy()
    ldf["label_time"] = pd.to_datetime(ldf["prediction_time"])
    ldf["date"] = ldf["label_time"].dt.date

    merged = ldf.merge(fdf, on=["subject_id", "date"], how="left")
    merged = merged[merged["feature_time"] < merged["label_time"]]

    if len(merged) == 0:
        # if feature_time equals label_time exactly, nudge features back by 1s
        fdf["feature_time"] = fdf["feature_time"] - pd.to_timedelta(1, unit="s")
        merged = ldf.merge(fdf, on=["subject_id", "date"], how="left")
        merged = merged[merged["feature_time"] < merged["label_time"]]
        if len(merged) == 0:
            raise RuntimeError("No label rows had a strictly earlier same-day feature state even after 1s nudge.")

    merged = merged.sort_values(["subject_id", "label_time", "feature_time"]).groupby(
        ["subject_id", "prediction_time", "label_time"], as_index=False
    ).tail(1)

    X = np.vstack([features["features"][i] for i in merged["feat_idx"].to_numpy()])
    return {
        "subject_ids": merged["subject_id"].to_numpy(),
        "prediction_times_label": merged["label_time"].to_numpy(),
        "prediction_times_feature": merged["feature_time"].to_numpy(),
        "y": merged["target_value"].to_numpy(dtype=float),
        "X": X,
    }


def main():
    args = create_arg_parser().parse_args()

    pretraining_data = pathlib.Path(args.pretraining_data)
    output_root = pathlib.Path(args.output_root) if args.output_root else pretraining_data

    labels_path = output_root / "labels"
    features_path = output_root / "features"
    results_root = output_root / "results"

    labels_to_run = LABEL_NAMES
    if args.cohort_label is not None:
        candidate = labels_path / (args.cohort_label + ".parquet")
        if candidate.exists():
            print(f"Using the user defined label at: {candidate}")
            labels_to_run = [args.cohort_label]
        else:
            raise RuntimeError(f"The user provided label does not exist at {candidate}")

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        for label_name in labels_to_run:
            label_output_dir = results_root / label_name / (f"motor_{args.observation_window}" if args.observation_window else "motor")
            label_output_dir.mkdir(exist_ok=True, parents=True)
            test_result_file = label_output_dir / "metrics.json"
            features_label_data = label_output_dir / "features_with_label"
            features_label_data.mkdir(exist_ok=True, parents=True)

            if test_result_file.exists():
                print(f"The result already existed for {label_name} at {test_result_file}, skipping.")
                continue

            labels = pd.read_parquet(labels_path / (label_name + ".parquet"))
            if not {"subject_id", "prediction_time", "target_value"}.issubset(labels.columns):
                raise RuntimeError("Label parquet must contain subject_id, prediction_time, target_value.")

            motor_features_name = get_motor_features_name(label_name, args.observation_window)
            with open(features_path / f"{motor_features_name}.pkl", "rb") as f:
                features = pickle.load(f)

            # keep only labels with subjects we have features for
            labels = labels[labels.subject_id.isin(set(features["subject_ids"]))].copy()
            labels = labels.sort_values(["subject_id", "prediction_time"])

            aligned = temporal_same_day_strict_future_join(features, labels)

            main_split = femr.splits.SubjectSplit.load_from_csv(args.main_split_path)
            subj = aligned["subject_ids"]
            train_mask = np.isin(subj, main_split.train_subject_ids)
            test_mask = np.isin(subj, main_split.test_subject_ids)

            X_train = aligned["X"][train_mask]
            y_train = aligned["y"][train_mask]
            X_test = aligned["X"][test_mask]
            y_test = aligned["y"][test_mask]

            pd.DataFrame({
                "subject_id": subj[train_mask],
                "prediction_time_label": aligned["prediction_times_label"][train_mask],
                "prediction_time_feature": aligned["prediction_times_feature"][train_mask],
                "target_value": y_train,
                "features": list(map(lambda v: v, X_train)),
            }).sample(frac=1.0, random_state=42).to_parquet(features_label_data / "train.parquet", index=False)

            pd.DataFrame({
                "subject_id": subj[test_mask],
                "prediction_time_label": aligned["prediction_times_label"][test_mask],
                "prediction_time_feature": aligned["prediction_times_feature"][test_mask],
                "target_value": y_test,
                "features": list(map(lambda v: v, X_test)),
            }).to_parquet(features_label_data / "test.parquet", index=False)

            print(f"Aligned total: {len(subj)}, train: {X_train.shape[0]}, test: {X_test.shape[0]}")

            model = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
            ])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            out_pred_dir = label_output_dir / "test_predictions"
            out_pred_dir.mkdir(exist_ok=True, parents=True)
            pl.DataFrame({
                "subject_id": subj[test_mask].tolist(),
                "prediction_time_feature": aligned["prediction_times_feature"][test_mask].tolist(),
                "prediction_time_label": aligned["prediction_times_label"][test_mask].tolist(),
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
                "residual": (y_test - y_pred).tolist(),
            }).write_parquet(out_pred_dir / "predictions.parquet")

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
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

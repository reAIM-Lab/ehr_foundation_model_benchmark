"""
Finetuning script (RidgeCV) with configurable alpha grid, shuffled CV, and baseline/label stats.
This does not modify the original finetune_regression.py behavior or outputs.

  - Narrow α grid: e.g., np.logspace(-2, 2, 9) to avoid extreme shrinkage if it helps performance.
  - Use shuffled CV: RidgeCV(cv=5) to stabilize α (default GCV can be brittle on some data).
  - Confirm label distributions: Compute train/test mean, std, and a mean-only baseline R2 to quantify
  lift.

- Added a new script: femr/omop_meds_tutorial/evaluation/regression/finetune_regression_cv.py
    - Alpha grid: defaults to logspace(-2, 2, 9) or pass --alphas "0.01,0.1,1,10,100".
    - CV: RidgeCV(cv=KFold(n_splits=5, shuffle=True/False, random_state=seed)).
    - Label stats: prints and saves train/test count, mean, std, min, max.
    - Baseline: mean predictor metrics (RMSE/MAE/R2) saved alongside model metrics.
    - Output isolation: writes under results/<task>/<model_name[_obs]_<suffix>> with default suffix cv.
    - Saves features_with_label/train.parquet and test.parquet, and test_predictions.parquet.

Example usage

- Single task:
    - python finetune_regression_cv.py --cohort_label pao2 \
--model_name motor --model_path /user/zj2398/cache/motor_mimic_8k/output/best_100620 --output_root /shared/share_mala/zj2398/mimic/regression/motor --main_split_path /user/zj2398/cache/motor_mimic_8k/main_split.csv \
--meds_reader /user/zj2398/cache/mimic/meds_v0.6_reader --cv_folds 5 --cv_shuffle --seed 42 --suffix cv
    - To customize alphas: add --alphas "0.01,0.0316,0.1,0.316,1,3.16,10,31.6,100".
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Optional

import meds_reader
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import femr.featurizers
import femr.splits

from femr.omop_meds_tutorial.evaluation.generate_motor_features import get_model_name
from femr.omop_meds_tutorial.evaluation.generate_labels import LABEL_NAMES


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RidgeCV finetuning with CV and baselines for numerical regression")

    parser.add_argument("--meds_reader", dest="meds_reader", default=None)
    parser.add_argument("--cohort_label", dest="cohort_label", default=None)
    parser.add_argument("--observation_window", dest="observation_window", type=int, default=None)
    parser.add_argument("--main_split_path", dest="main_split_path", default=None)
    parser.add_argument("--model_name", dest="model_name", required=True)
    parser.add_argument("--model_path", dest="model_path", required=True)
    parser.add_argument("--output_root", dest="output_root", required=True)

    # New options
    parser.add_argument(
        "--alphas",
        dest="alphas",
        default="",
        help="Comma-separated list of alphas. If empty, uses logspace(-2, 2, 9)",
    )
    parser.add_argument("--cv_folds", dest="cv_folds", type=int, default=5)
    parser.add_argument("--cv_shuffle", dest="cv_shuffle", action="store_true")
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument(
        "--suffix",
        dest="suffix",
        default="cv",
        help="Suffix to append to model_name in the results directory to avoid overwrites",
    )

    return parser


def parse_alphas(alpha_str: str) -> np.ndarray:
    if alpha_str is None or alpha_str.strip() == "":
        return np.logspace(-2, 2, 9)
    parts = [p.strip() for p in alpha_str.split(",") if p.strip()]
    values: List[float] = []
    for p in parts:
        try:
            values.append(float(p))
        except ValueError:
            raise ValueError(f"Invalid alpha value: {p}")
    if len(values) == 0:
        return np.logspace(-2, 2, 9)
    return np.asarray(values, dtype=float)


def main() -> None:
    args = create_arg_parser().parse_args()
    output_root = Path(args.output_root)

    # Resolve which label(s) to run
    label_names: List[str] = LABEL_NAMES
    if args.cohort_label is not None:
        label_path = output_root / "labels" / f"{args.cohort_label}.parquet"
        if not label_path.exists():
            raise RuntimeError(f"The user provided label does not exist at {label_path}")
        label_names = [args.cohort_label]

    # Prepare alpha grid and CV
    alphas = parse_alphas(args.alphas)
    kf = KFold(n_splits=args.cv_folds, shuffle=args.cv_shuffle, random_state=(args.seed if args.cv_shuffle else None))

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=32):
        for label_name in label_names:
            # Results directory: append suffix to model_name to avoid overwriting original runs
            if args.observation_window:
                suffix_name = f"{args.model_name}_{args.observation_window}_{args.suffix}"
            else:
                suffix_name = f"{args.model_name}_{args.suffix}"

            label_output_dir = output_root / "results" / label_name / suffix_name
            label_output_dir.mkdir(exist_ok=True, parents=True)

            test_result_file = label_output_dir / "metrics.json"
            features_label_data = label_output_dir / "features_with_label"
            features_label_data.mkdir(exist_ok=True, parents=True)

            # Load label table
            labels_df = pd.read_parquet(output_root / "labels" / f"{label_name}.parquet")

            # Load features
            motor_features_name = get_model_name(label_name, args.model_name, args.observation_window)
            features_path = output_root / "features"
            with open(features_path / f"{motor_features_name}.pkl", "rb") as f:
                features = pickle.load(f)

            # Filter labels to those that have features; sort consistently
            labels_with_features = labels_df[labels_df.subject_id.isin(features["subject_ids"])].copy()
            labels_with_features = labels_with_features.sort_values(["subject_id", "prediction_time"])

            # Join numeric labels to features
            labeled_features = femr.featurizers.join_labels_numerical(features, labels_with_features)

            # Subject split
            main_split = femr.splits.SubjectSplit.load_from_csv(args.main_split_path)
            train_mask = np.isin(labeled_features["subject_ids"], main_split.train_subject_ids)
            test_mask = np.isin(labeled_features["subject_ids"], main_split.test_subject_ids)

            def apply_mask(values, mask):
                def apply(_k, v):
                    if len(v.shape) == 1:
                        return v[mask]
                    elif len(v.shape) == 2:
                        return v[mask, :]
                    else:
                        raise AssertionError(f"Cannot handle shape {v.shape}")

                return {k: apply(k, v) for k, v in values.items()}

            train_data = apply_mask(labeled_features, train_mask)
            test_data = apply_mask(labeled_features, test_mask)

            # Persist train/test rows for reproducibility
            train_features_list = [feat for feat in train_data["features"]]
            train_set = pd.DataFrame(
                {
                    "subject_id": train_data["subject_ids"],
                    "prediction_time": train_data["prediction_times"],
                    "numerical_value": train_data["numerical_value"],
                    "features": train_features_list,
                }
            ).sample(frac=1.0, random_state=args.seed)
            train_set.to_parquet(features_label_data / "train.parquet", index=False)

            test_features_list = [feat for feat in test_data["features"]]
            pd.DataFrame(
                {
                    "subject_id": test_data["subject_ids"],
                    "prediction_time": test_data["prediction_times"],
                    "numerical_value": test_data["numerical_value"],
                    "features": test_features_list,
                }
            ).to_parquet(features_label_data / "test.parquet", index=False)

            # Basic dataset info
            print(f"Total labels: {len(labels_df)}")
            print(f"Total train features labels: {len(train_data['features'])}")
            print(f"Total test features labels: {len(test_data['features'])}")

            y_train = train_data["numerical_value"].astype(float)
            y_test = test_data["numerical_value"].astype(float)

            # Label stats
            label_stats = {
                "train": {
                    "count": int(y_train.shape[0]),
                    "mean": float(np.mean(y_train)) if y_train.size else float("nan"),
                    "std": float(np.std(y_train)) if y_train.size else float("nan"),
                    "min": float(np.min(y_train)) if y_train.size else float("nan"),
                    "max": float(np.max(y_train)) if y_train.size else float("nan"),
                },
                "test": {
                    "count": int(y_test.shape[0]),
                    "mean": float(np.mean(y_test)) if y_test.size else float("nan"),
                    "std": float(np.std(y_test)) if y_test.size else float("nan"),
                    "min": float(np.min(y_test)) if y_test.size else float("nan"),
                    "max": float(np.max(y_test)) if y_test.size else float("nan"),
                },
            }
            print(
                f"Train y: mean={label_stats['train']['mean']:.3f}, std={label_stats['train']['std']:.3f}; "
                f"Test y: mean={label_stats['test']['mean']:.3f}, std={label_stats['test']['std']:.3f}"
            )

            # Baseline (mean of y_train)
            baseline_mean = float(np.mean(y_train)) if y_train.size else 0.0
            y_pred_base = np.full_like(y_test, baseline_mean, dtype=float)
            baseline_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_base)))
            baseline_mae = float(mean_absolute_error(y_test, y_pred_base))
            baseline_r2 = float(r2_score(y_test, y_pred_base))

            # Model: StandardScaler + RidgeCV with custom alphas and KFold CV
            model = Pipeline(
                [
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    ("ridge", RidgeCV(alphas=alphas, cv=kf)),
                ]
            )

            # Use array inputs directly to avoid dtype surprises
            X_train = train_data["features"].astype(np.float32, copy=False)
            X_test = test_data["features"].astype(np.float32, copy=False)

            print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, alpha grid: {alphas.tolist()}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Save predictions
            pl.DataFrame(
                {
                    "subject_id": test_data["subject_ids"].tolist(),
                    "prediction_time": test_data["prediction_times"].tolist(),
                    "y_true": y_test.tolist(),
                    "y_pred": y_pred.tolist(),
                    "residual": (y_test - y_pred).tolist(),
                }
            ).write_parquet(label_output_dir / "test_predictions.parquet")

            # Model metrics
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            metrics = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "alpha": float(model.named_steps["ridge"].alpha_),
                "alphas": alphas.tolist(),
                "cv_folds": args.cv_folds,
                "cv_shuffle": bool(args.cv_shuffle),
                "seed": args.seed,
                "label_stats": label_stats,
                "baseline": {
                    "mean": baseline_mean,
                    "rmse": baseline_rmse,
                    "mae": baseline_mae,
                    "r2": baseline_r2,
                },
            }

            print(f"{label_name} ridge α={metrics['alpha']:.4g} | RMSE={rmse:.4g} | MAE={mae:.4g} | R2={r2:.4g} |baseline R2={baseline_r2:.4g}")
            with open(test_result_file, "w") as f:
                json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()


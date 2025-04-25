import argparse
from pathlib import Path

import numpy as np
import polars as pl
import json
import pickle
from meds import train_split, tuning_split, held_out_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

TRAIN_SIZES = [100, 1000, 10000, 100000]


def main(args):
    meds_dir = Path(args.meds_dir)
    subject_splits_path = meds_dir / "metadata" / "subject_splits.parquet"
    print(f"Loading subject_splits.parquet from {subject_splits_path}")
    subject_splits = pl.read_parquet(subject_splits_path)
    features_label_input_dir = Path(args.features_label_input_dir)
    features_label = pl.read_parquet(list(features_label_input_dir.rglob('*.parquet')))

    output_dir = Path(args.output_dir)
    task_output_dir = output_dir / args.task_name
    task_output_dir.mkdir(exist_ok=True, parents=True)

    features_label = features_label.sort("subject_id", "prediction_time")

    train_dataset = features_label.join(
        subject_splits.select("subject_id", "split"), "subject_id"
    ).filter(
        pl.col("split").is_in([train_split, tuning_split])
    )
    test_dataset = features_label.join(
        subject_splits.select("subject_id", "split"), "subject_id"
    ).filter(
        pl.col("split") == held_out_split
    )

    for size in TRAIN_SIZES:
        few_show_output_dir = task_output_dir / f"results_{size}"
        few_show_output_dir.mkdir(exist_ok=True, parents=True)
        logistic_model_file = few_show_output_dir / "model.pickle"
        logistic_test_result_file = few_show_output_dir / "metrics.json"
        if logistic_test_result_file.exists():
            print(
                f"The results for logistic regression with {size} shots already exist at {logistic_test_result_file}"
            )
        else:
            if size < 100000:
                subset = train_dataset.sample(n=size, shuffle=True, seed=args.seed)
            else:
                subset = train_dataset

            if logistic_model_file.exists():
                print(
                    f"The logistic regression model already exist for {size} shots, loading it from {logistic_model_file}"
                )
                with open(logistic_model_file, "rb") as f:
                    model = pickle.load(f)
            else:
                model = LogisticRegressionCV(scoring="roc_auc")
                model.fit(np.asarray(subset["features"].to_list()), subset["boolean_value"].to_numpy())
                with open(logistic_model_file, "wb") as f:
                    pickle.dump(model, f)

            y_pred = model.predict_proba(test_dataset["features"].to_numpy())[:, 1]
            logistic_predictions = pl.DataFrame(
                {
                    "subject_id": test_dataset["subject_id"].to_list(),
                    "prediction_time": test_dataset["prediction_time"].to_list(),
                    "predicted_boolean_probability": y_pred,
                    "predicted_boolean_value": None,
                    "boolean_value": test_dataset["boolean_value"].cast(pl.Boolean).to_list(),
                }
            )
            logistic_predictions = logistic_predictions.with_columns(
                pl.col("predicted_boolean_value").cast(pl.Boolean())
            )
            logistic_test_predictions = few_show_output_dir / "test_predictions"
            logistic_test_predictions.mkdir(exist_ok=True, parents=True)
            logistic_predictions.write_parquet(
                logistic_test_predictions / "predictions.parquet"
            )
            roc_auc = roc_auc_score(test_dataset["boolean_value"], y_pred)
            precision, recall, _ = precision_recall_curve(
                test_dataset["boolean_value"], y_pred
            )
            pr_auc = auc(recall, precision)
            metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc}
            print("Logistic:", size, args.task_name, metrics)
            with open(logistic_test_result_file, "w") as f:
                json.dump(metrics, f, indent=4)


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

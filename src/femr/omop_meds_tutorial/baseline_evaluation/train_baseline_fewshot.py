import femr.splits
from pathlib import Path

import pandas as pd
import polars as pl
import femr.featurizers
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack
import optuna
import functools
import lightgbm as lgb
from .train_baseline import create_arg_parser, lightgbm_objective, get_baseline_features_name, save_to_json
from meds import train_split as meds_train_split, tuning_split, held_out_split
import pickle

SEED = 42
MINIMUM_NUM_CASES = 10
TRAIN_SIZES = [100, 1000, 10000, 100000]


def main():
    args = create_arg_parser().parse_args()
    meds_dir = Path(args.meds_reader)
    subject_splits_path = meds_dir / "metadata" / "subject_splits.parquet"
    print(f"Loading subject_splits.parquet from {subject_splits_path}")
    subject_splits = pl.read_parquet(subject_splits_path)
    models_path = Path(args.pretraining_data) / "models"
    models_path.mkdir(exist_ok=True)

    label_path = models_path.parent / "labels" / (args.cohort_label + '.parquet')
    if label_path.exists():
        print(f"Using the user defined label at: {label_path}")
    else:
        raise RuntimeError(f"The user provided label does not exist at {label_path}")

    output_dir = models_path.parent / "results"
    task_output_dir = output_dir / args.cohort_label
    task_output_dir.mkdir(exist_ok=True, parents=True)

    if args.observation_window:
        label_output_dir = output_dir / args.cohort_label / f"baseline_{args.observation_window}"
    else:
        label_output_dir = output_dir / args.cohort_label / f"baseline"
    label_output_dir.mkdir(exist_ok=True, parents=True)

    done_file = task_output_dir / "done"
    if done_file.exists():
        print(f"The results for {args.cohort_label} already exist because the indicator file is present at {done_file}")
        exit(0)

    labels = pd.read_parquet(models_path.parent / "labels" / (args.cohort_label + '.parquet'))
    labels = labels.sort_values(["subject_id", "prediction_time"])

    with open(models_path.parent / 'features' / get_baseline_features_name(args.cohort_label, args.observation_window),
              'rb') as f:
        features = pickle.load(f)

    labeled_features = femr.featurizers.join_labels(features, labels)

    train_features_list = [feature for feature in labeled_features["features"]]
    all_features_with_label = pl.DataFrame({
        "subject_id": labeled_features["subject_ids"].tolist(),
        "prediction_time": labeled_features["prediction_times"].tolist(),
        "features": train_features_list,
        "boolean_value": labeled_features["boolean_values"],
    }).with_row_index(
        name="sample_id",
        offset=1
    )

    # The test labels need to be in a pandas dataframe
    train_labels = all_features_with_label.join(
        subject_splits.select("subject_id", "split"), "subject_id"
    ).filter(
        pl.col("split").is_in([meds_train_split, tuning_split])
    )

    held_out_subject_ids = subject_splits.filter(pl.col("split").eq(held_out_split))["subject_id"].to_list()
    # The test labels need to be in a pandas dataframe
    test_labels = labels[labels.subject_id.isin(held_out_subject_ids)]
    test_data = femr.featurizers.join_labels(features, test_labels)

    should_terminate = False
    # We keep track of the sample ids that have been picked from the previous few-shots experiments.
    existing_sample_ids = set()
    for size in TRAIN_SIZES:
        # This indicates the data set has reached its maximum size, and we should terminate
        if should_terminate:
            break

        if len(train_labels) < size:
            size = len(train_labels)
            should_terminate = True

        gbm_parquet_file = task_output_dir / f"gbm_{size}.parquet"
        gbm_output_dir = task_output_dir / f"gbm_{size}"
        gbm_output_dir.mkdir(exist_ok=True, parents=True)
        gbm_test_metrics_file = gbm_output_dir / "metrics.json"

        logistic_parquet_file = task_output_dir / f"logistic_{size}.parquet"
        logistic_output_dir = task_output_dir / f"logistic_{size}"
        logistic_output_dir.mkdir(exist_ok=True, parents=True)
        logistic_test_metrics_file = logistic_output_dir / "metrics.json"

        if logistic_test_metrics_file.exists() and gbm_test_metrics_file.exists():
            print(f"The results already exist for {size} shots experiment")
            continue

        remaining_train_labels = train_labels.filter(~pl.col("sample_id").is_in(existing_sample_ids))
        existing_samples = train_labels.filter(pl.col("sample_id").is_in(existing_sample_ids))
        try:
            size_required = size - len(existing_samples)
            success = True
            subset = pl.concat([
                remaining_train_labels.sample(n=size_required, seed=SEED),
                existing_samples
            ]).sample(
                fraction=1.0,
                shuffle=True,
                seed=SEED
            )
            while True:
                count_by_class = subset.group_by("boolean_value").count().to_dict(as_series=False)
                if len(count_by_class["boolean_value"]) == 1:
                    success = False
                else:
                    for cls, count in zip(count_by_class["boolean_value"], count_by_class["count"]):
                        if cls == 1 and count < MINIMUM_NUM_CASES:
                            success = False
                            print(f"The number of positive cases is less than {MINIMUM_NUM_CASES} for {size}")
                            break
                if success:
                    break
                else:
                    n_positive_cases = len(subset.filter(pl.col("boolean_value") == True))
                    sampling_percentage = size_required / len(remaining_train_labels)
                    n_positives_to_sample = max(MINIMUM_NUM_CASES, int(n_positive_cases * sampling_percentage))
                    positives_subset = remaining_train_labels.filter(pl.col("boolean_value") == True).sample(
                        n=n_positives_to_sample, shuffle=True, seed=SEED, with_replacement=True
                    )
                    negatives_subset = remaining_train_labels.filter(pl.col("boolean_value") == False).sample(
                        n=(size_required - n_positives_to_sample), shuffle=True, seed=SEED
                    )
                    print(
                        f"number of positive cases: {len(positives_subset)}; "
                        f"number of negative cases: {len(negatives_subset)}"
                    )
                    subset = pl.concat([positives_subset, negatives_subset]).sample(
                        fraction=1.0,
                        shuffle=True,
                        seed=SEED
                    )
                    break

            existing_sample_ids.update(subset["sample_id"].to_list())

            # The following logic requires a pandas dataframe
            subset = subset.rename({
                "subject_id": "subject_ids",
                "prediction_time": "prediction_times",
                "boolean_value" : "boolean_values",
            })
            if gbm_test_metrics_file.exists():
                print(
                    f"The result already exists for GBM {args.cohort_label} "
                    f"at {gbm_test_metrics_file}, it will be skipped!"
                )
            else:
                try:
                    print(f"Starting training GBM for {size}")
                    # Split data into training and testing sets
                    train_data, dev_data = train_test_split(subset, test_size=0.1, random_state=17)
                    lightgbm_study = optuna.create_study()  # Create a new study.
                    lightgbm_study.optimize(
                        functools.partial(
                            lightgbm_objective,
                            train_data={
                                "features" : vstack(train_data["features"].to_list()),
                                "boolean_values": train_data["boolean_values"].to_numpy(),
                            },
                            dev_data={
                                "features" : vstack(dev_data["features"].to_list()),
                                "boolean_values": dev_data["boolean_values"].to_numpy(),
                            }
                        ),
                        n_trials=10
                    )
                    print(f"Computing predictions for gbm for {size}")
                    best_num_trees = lightgbm_study.best_trial.user_attrs['num_trees']
                    best_params = lightgbm_study.best_trial.params
                    best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
                    final_train_data = {
                        "features" : vstack(train_data["features"].to_list()),
                        "boolean_values": train_data["boolean_values"].to_numpy(),
                    }
                    dtrain_final = lgb.Dataset(
                        final_train_data["features"],
                        label=final_train_data["boolean_values"],
                    )
                    gbm_final = lgb.train(best_params, dtrain_final, num_boost_round=best_num_trees)

                    # Generate predictions on test data.
                    lightgbm_preds = gbm_final.predict(test_data['features'], raw_score=False)
                    final_lightgbm_auroc = lightgbm_objective(
                        lightgbm_study.best_trial,
                        train_data=final_train_data,
                        dev_data=test_data,
                        num_trees=lightgbm_study.best_trial.user_attrs['num_trees']
                    )
                    print("gbm", args.cohort_label, final_lightgbm_auroc)
                    lightgbm_results = {
                        "label_name": args.cohort_label,
                        "final_lightgbm_auroc": final_lightgbm_auroc,
                    }
                    save_to_json(lightgbm_results, gbm_test_metrics_file)
                    lightgbm_predictions = pl.DataFrame({
                        "subject_id": test_data["subject_ids"].tolist(),
                        "prediction_time": test_data["prediction_times"].tolist(),
                        "predicted_boolean_probability": lightgbm_preds.tolist(),
                        "predicted_boolean_value": None,
                        "boolean_value": test_data["boolean_values"].astype(bool).tolist()
                    })
                    lightgbm_predictions = lightgbm_predictions.with_columns(
                        pl.col("predicted_boolean_value").cast(pl.Boolean())
                    )
                    lightgbm_predictions.write_parquet(gbm_parquet_file)
                except Exception as e:
                    print(e)

            if logistic_test_metrics_file.exists():
                print(
                    f"The result already exists for Logistic {args.cohort_label} "
                    f"at {logistic_test_metrics_file}, it will be skipped!"
                )
            else:
                print(f"Starting training logistic for {size}")
                logistic_model = LogisticRegressionCV(scoring='roc_auc', random_state=42, max_iter=500)
                logistic_model.fit(
                    vstack(subset['features'].to_list()),
                    subset['boolean_values'].to_numpy()
                )
                logistic_y_pred = logistic_model.predict_proba(test_data['features'])[:, 1]
                final_logistic_auroc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], logistic_y_pred)
                print('logistic', final_logistic_auroc, args.cohort_label)
                logistic_results = {
                    "label_name": args.cohort_label,
                    "final_logistic_auroc": final_logistic_auroc
                }
                save_to_json(logistic_results, logistic_test_metrics_file)
                logistic_predictions = pl.DataFrame({
                    "subject_id": test_data["subject_ids"].tolist(),
                    "prediction_time": test_data["prediction_times"].tolist(),
                    "predicted_boolean_probability": logistic_y_pred.tolist(),
                    "predicted_boolean_value": None,
                    "boolean_value": test_data["boolean_values"].astype(bool).tolist()
                })
                logistic_predictions = logistic_predictions.with_columns(
                    pl.col("predicted_boolean_value").cast(pl.Boolean())
                )
                logistic_predictions.write_parquet(logistic_parquet_file)
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    main()

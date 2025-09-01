"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""

import femr.splits
import meds_reader
import pandas as pd
import polars as pl
import femr.featurizers
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegressionCV
import optuna
import functools
import lightgbm as lgb
from .generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser
from .generate_tabular_features import get_baseline_features_name

import pickle
import json


def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def save_to_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def lightgbm_objective(trial, *, train_data, dev_data, num_trees=None):
    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,

        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    dtrain = lgb.Dataset(train_data['features'], label=train_data['boolean_values'])
    ddev = lgb.Dataset(dev_data['features'], label=dev_data['boolean_values'])

    if num_trees is None:
        callbacks = [lgb.early_stopping(10)]
        gbm = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=(ddev,), callbacks=callbacks)
    else:
        gbm = lgb.train(param, dtrain, num_boost_round=num_trees)

    y_pred = gbm.predict(dev_data['features'], raw_score=True)

    error = -sklearn.metrics.roc_auc_score(dev_data['boolean_values'], y_pred)

    if num_trees is None:
        trial.set_user_attr("num_trees", gbm.best_iteration + 1)

    return error


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
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
    return args


def main():
    from pathlib import Path
    args = create_arg_parser().parse_args()
    models_path = Path(args.pretraining_data) / "models"
    models_path.mkdir(exist_ok=True)

    labels = LABEL_NAMES
    if args.cohort_label is not None:
        label_path = models_path.parent / "labels" / (args.cohort_label + '.parquet')
        if label_path.exists():
            print(f"Using the user defined label at: {label_path}")
            labels = [args.cohort_label]
        else:
            raise RuntimeError(f"The user provided label does not exist at {label_path}")

    output_dir = models_path.parent / "results"
    output_dir.mkdir(exist_ok=True, parents=True)
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        for label_name in labels:
            if args.observation_window:
                label_output_dir = output_dir / label_name / f"baseline_{args.observation_window}"
            else:
                label_output_dir = output_dir / label_name / f"baseline"

            label_output_dir.mkdir(exist_ok=True, parents=True)
            done_file = label_output_dir / "done"
            if done_file.exists():
                print(f"The results for {label_name} already exist because the indicator file is present at {done_file}")
                continue

            labels = pd.read_parquet(models_path.parent / "labels" / (label_name + '.parquet'))
            labels = labels.sort_values(["subject_id", "prediction_time"])
            labels = labels.sample(n=len(labels), random_state=42, replace=False)
            with open(models_path.parent / 'features' / get_baseline_features_name(label_name, args.observation_window), 'rb') as f:
                features = pickle.load(f)

            # Remove the labels that do not have features generated
            labels = labels[labels.subject_id.isin(features["subject_ids"])]
            labels = labels.sort_values(["subject_id", "prediction_time"])

            labeled_features = femr.featurizers.join_labels(features, labels)
            main_split = femr.splits.SubjectSplit.load_from_csv(str(models_path.parent / 'main_split.csv'))
            train_split = femr.splits.generate_hash_split(main_split.train_subject_ids, 17, frac_test=0.10)

            train_mask = np.isin(labeled_features['subject_ids'], train_split.train_subject_ids)
            dev_mask = np.isin(labeled_features['subject_ids'], train_split.test_subject_ids)
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
            dev_data = apply_mask(labeled_features, dev_mask)
            test_data = apply_mask(labeled_features, test_mask)

            gbm_output_dir = label_output_dir / "gbm"
            gbm_output_dir.mkdir(exist_ok=True, parents=True)

            gbm_metrics_output_file = gbm_output_dir / f'metrics.json'
            if gbm_metrics_output_file.exists():
                print(
                    f"The result already exists for GBM {label_name} at {gbm_metrics_output_file}, it will be skipped!")
            else:
                try:
                    lightgbm_study = optuna.create_study()  # Create a new study.
                    lightgbm_study.optimize(functools.partial(lightgbm_objective, train_data=train_data, dev_data=dev_data),
                                            n_trials=10)  # Invoke optimization of the objective function.

                    final_train_data = apply_mask(labeled_features, train_mask | dev_mask)
                    print("Computing predictions")
                    best_num_trees = lightgbm_study.best_trial.user_attrs['num_trees']
                    best_params = lightgbm_study.best_trial.params
                    best_params.update({"objective": "binary", "metric": "auc", "verbosity": -1})
                    dtrain_final = lgb.Dataset(final_train_data['features'], label=final_train_data['boolean_values'])
                    gbm_final = lgb.train(best_params, dtrain_final, num_boost_round=best_num_trees)

                    # Generate predictions on test data.
                    lightgbm_preds = gbm_final.predict(test_data['features'], raw_score=False)
                    final_lightgbm_auroc2 = -sklearn.metrics.roc_auc_score(test_data['boolean_values'], lightgbm_preds)

                    final_lightgbm_auroc = lightgbm_objective(lightgbm_study.best_trial, train_data=final_train_data,
                                                              dev_data=test_data,
                                                              num_trees=lightgbm_study.best_trial.user_attrs['num_trees'])
                    print(label_name)

                    print("Saving predictions")

                    print('lightgbm', final_lightgbm_auroc, label_name)
                    lightgbm_results = {
                        "label_name": label_name,
                        "final_lightgbm_auroc": final_lightgbm_auroc,
                        "final_lightgbm_auroc2": final_lightgbm_auroc2,
                    }

                    save_to_json(lightgbm_results, gbm_metrics_output_file)
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
                    gbm_test_predictions = gbm_output_dir / "test_predictions"
                    gbm_test_predictions.mkdir(exist_ok=True, parents=True)
                    lightgbm_predictions.write_parquet(gbm_test_predictions / "test_gbm_predictions.parquet")
                except Exception as e:
                    print(e)

            logistic_output_dir = label_output_dir / "logistic"
            logistic_metrics_output_file = logistic_output_dir / f'metrics.json'
            if logistic_metrics_output_file.exists():
                print(
                    f"The result already exists for Logistic {label_name} at {logistic_metrics_output_file}, it will be skipped!")
            else:
                final_train_data = apply_mask(labeled_features, train_mask | dev_mask)
                logistic_output_dir.mkdir(exist_ok=True, parents=True)
                logistic_model = LogisticRegressionCV(scoring='roc_auc')
                logistic_model.fit(final_train_data['features'], final_train_data['boolean_values'])
                logistic_y_pred = logistic_model.predict_proba(test_data['features'])[:, 1]
                final_logistic_auroc = sklearn.metrics.roc_auc_score(test_data['boolean_values'], logistic_y_pred)
                print('logistic', final_logistic_auroc, label_name)
                logistic_results = {
                    "label_name": label_name,
                    "final_logistic_auroc": final_logistic_auroc
                }

                save_to_json(logistic_results, logistic_metrics_output_file)

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
                logistic_test_predictions = logistic_output_dir / "test_predictions"
                logistic_test_predictions.mkdir(exist_ok=True, parents=True)
                logistic_predictions.write_parquet(logistic_test_predictions / "predictions.parquet")

            try:
                f = open(done_file, "x")
            except FileExistsError:
                print("File already exists.")
            finally:
                f.close()

if __name__ == "__main__":
    main()

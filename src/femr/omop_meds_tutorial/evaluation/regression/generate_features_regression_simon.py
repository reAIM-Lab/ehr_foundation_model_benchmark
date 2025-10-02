# femr/omop_meds_tutorial/motor_evaluation/generate_motor_features_regression.py

import os
import sys
import glob
import json
import datetime
from typing import Optional

import argparse
import pandas as pd
import numpy as np
import pickle
import pathlib
import torch
import meds
import meds_reader
import femr.transforms
# import femr.models.transformer
import femr.models.architecture.embedding


from femr.omop_meds_tutorial.motor_evaluation.generate_labels import (  # type: ignore
        create_omop_meds_tutorial_arg_parser,
        LABEL_NAMES,
    )

# Robust import so file or -m both work
# try:
#     from ..generate_labels import create_omop_meds_tutorial_arg_parser, LABEL_NAMES
# except Exception:
#     _here = pathlib.Path(__file__).resolve()
#     _repo_root = _here.parents[3]  # path that CONTAINS 'femr'
#     if str(_repo_root) not in sys.path:
#         sys.path.insert(0, str(_repo_root))
#     from femr.omop_meds_tutorial.motor_evaluation.generate_labels import (  # type: ignore
#         create_omop_meds_tutorial_arg_parser,
#         LABEL_NAMES,
#     )


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument("--num_proc", type=int, default=6)
    args.add_argument("--model_path", default=None)
    args.add_argument("--device", default="cuda")
    args.add_argument("--tokens_per_batch", type=int, default=8192)
    args.add_argument("--cohort_dir", default=None)
    args.add_argument("--observation_window", type=int, default=None)
    args.add_argument("--min_subjects_per_batch", type=int, default=1)
    args.add_argument("--ontology_path", default=None)
    args.add_argument("--linear_interpolation", dest="use_linear_interpolation", action="store_true")
    args.add_argument(
        "--loss_type",
        dest="loss_type",
        default=None,
        help="The loss type",
    )
    args.add_argument(
        "--output_root",
        default=None,
        help="Directory where labels/features/flops will be written. Defaults to --pretraining_data.",
    )
    return args


def read_recursive_parquet(root_dir: str) -> pd.DataFrame:
    all_files = glob.glob(os.path.join(root_dir, "**", "*.parquet"), recursive=True)
    for p in all_files:
        print(p)
    if len(all_files) == 0:
        raise RuntimeError(f"No parquet files found under {root_dir}")
    return pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)


def coerce_to_regression_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to columns: subject_id, prediction_time, target_value.
    Accept aliases: patient_id -> subject_id, time -> prediction_time, numeric_value -> target_value.
    """
    df = df.copy()
    colmap = {}
    if "subject_id" in df.columns:
        colmap["subject_id"] = "subject_id"
    elif "patient_id" in df.columns:
        colmap["patient_id"] = "subject_id"
    else:
        raise RuntimeError("Missing subject identifier: subject_id/patient_id")

    if "prediction_time" in df.columns:
        colmap["prediction_time"] = "prediction_time"
    elif "time" in df.columns:
        colmap["time"] = "prediction_time"
    else:
        raise RuntimeError("Missing time column: prediction_time/time")

    if "target_value" in df.columns:
        colmap["target_value"] = "target_value"
    elif "numeric_value" in df.columns:
        colmap["numeric_value"] = "target_value"
    else:
        raise RuntimeError("Missing target column: target_value/numeric_value")

    df = df.rename(columns=colmap)
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    df["target_value"] = pd.to_numeric(df["target_value"], errors="coerce").astype(float)
    df = df.dropna(subset=["target_value"])
    return df[["subject_id", "prediction_time", "target_value"]]


def get_motor_features_name(label_name: str, observation_window: Optional[int] = None) -> str:
    return f"{label_name}_motor_{observation_window}" if observation_window else f"{label_name}_motor"


def _build_feature_times_in_return_order(features: dict, label_df: pd.DataFrame) -> np.ndarray:
    """
    Given returned `features["subject_ids"]` and the original label_df,
    construct a vector of prediction_times aligned to the returned feature order.
    We create a FIFO queue of times per subject (sorted), then iterate the
    returned subject_ids in order, popping the next time for that subject.
    """
    # queues of times per subject, ordered by prediction_time
    ldf = label_df.sort_values(["subject_id", "prediction_time"])
    queues: dict[int, list[pd.Timestamp]] = {}
    for sid, group in ldf.groupby("subject_id", sort=True):
        queues[int(sid)] = list(group["prediction_time"].tolist())

    aligned_times: list[pd.Timestamp] = []
    missing = 0
    for sid in features["subject_ids"]:
        sid_int = int(sid)
        q = queues.get(sid_int, [])
        if not q:
            # No remaining label times for this subject; bail out to a safe default
            missing += 1
            aligned_times.append(pd.NaT)
        else:
            aligned_times.append(q.pop(0))

    if missing > 0:
        # If we had NaT slots, try to backfill by using the last seen time per subject.
        last_seen: dict[int, pd.Timestamp] = {}
        for i, sid in enumerate(features["subject_ids"]):
            sid_int = int(sid)
            if pd.isna(aligned_times[i]):
                if sid_int in last_seen:
                    aligned_times[i] = last_seen[sid_int]
            else:
                last_seen[sid_int] = aligned_times[i]
        # Still NaTs? Final fallback: drop them at finetune time.

    return np.array(aligned_times, dtype="datetime64[ns]")


def main():
    args = create_arg_parser().parse_args()

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        pretraining_data = pathlib.Path(args.pretraining_data)
        output_root = pathlib.Path(args.output_root) if args.output_root else pretraining_data
        ontology_path = args.ontology_path

        features_path = output_root / "features"
        features_path.mkdir(exist_ok=True, parents=True)
        flops_path = output_root / "flops"
        flops_path.mkdir(exist_ok=True, parents=True)
        labels_path = output_root / "labels"
        labels_path.mkdir(exist_ok=True, parents=True)

        with open(ontology_path, "rb") as f:
            ontology = pickle.load(f)

        labels = LABEL_NAMES
        if args.cohort_dir is not None:
            if os.path.isdir(args.cohort_dir):
                label_name = os.path.basename(os.path.normpath(args.cohort_dir))
                print(f"label_name of cohort_dir: {label_name}")
                cohort = read_recursive_parquet(args.cohort_dir)
            else:
                label_name = os.path.basename(os.path.splitext(args.cohort_dir)[0])
                ext = os.path.splitext(args.cohort_dir)[1].lower()
                if ext == ".parquet":
                    cohort = pd.read_parquet(args.cohort_dir)
                elif ext == ".csv":
                    cohort = pd.read_csv(args.cohort_dir)
                else:
                    raise RuntimeError(f"Unknown file extension: {ext}")

            cohort = coerce_to_regression_labels(cohort)
            cohort.to_parquet(labels_path / (label_name + ".parquet"), index=False)
            # labels = [label_name]

        # for label_name in labels:
        motor_features_name = get_motor_features_name(label_name, args.observation_window)
        feature_output_path = features_path / f"{motor_features_name}.pkl"
        training_metrics_file = flops_path / f"{motor_features_name}.json"
        if feature_output_path.exists():
            print(f"The features for {label_name} already exist at {feature_output_path}, skipping.")
            continue

        file_path = labels_path / (label_name + ".parquet")
        print("Loading regression labels from ", file_path)
        label_df = pd.read_parquet(file_path)
        print(f"labels head:\n{label_df.head()}")

        typed_labels = [
            meds.Label(
                subject_id=label["subject_id"],
                prediction_time=label["prediction_time"],
                boolean_value=None,  # not used for feature extraction
            )
            for label in label_df.to_dict(orient="records")
        ]
        print(f"typed_labels length: {len(typed_labels)}")

        # try:
        #     total_flops = femr.models.transformer.TotalFlops()
        # except Exception:
        #     class _Dummy:
        #         total_flops = None
        #     total_flops = _Dummy()
        start_time: datetime.datetime = datetime.datetime.now()

        base = os.path.basename(pretraining_data)
        use_linear = args.use_linear_interpolation
        features = femr.models.transformer.compute_features(
            db=database,
            model_path=args.model_path,
            labels=typed_labels,
            ontology=ontology,
            device=torch.device(args.device),
            tokens_per_batch=args.tokens_per_batch,
            num_proc=args.num_proc,
            observation_window=args.observation_window,
            min_subjects_per_batch=args.min_subjects_per_batch,
            loss_type=args.loss_type
        )

        # if base in {"motor_mimic_bin_8_start_idx_corrected", "motor_mimic_bin_8_linear_interpolation"}:
        #     print("load from start_idx or interpolation")
        #     features = femr.models.transformer_linear_interpolation.compute_features(
        #         db=database,
        #         model_path=args.model_path,
        #         labels=typed_labels,
        #         ontology=ontology,
        #         device=torch.device(args.device),
        #         tokens_per_batch=args.tokens_per_batch,
        #         num_proc=args.num_proc,
        #         observation_window=args.observation_window,
        #         min_subjects_per_batch=args.min_subjects_per_batch,
        #         use_linear_interpolation=use_linear,
        #     )
        # else:
        #     print("load from normal model")
        #     features = femr.models.transformer.compute_features(
        #         db=database,
        #         model_path=args.model_path,
        #         labels=typed_labels,
        #         ontology=ontology,
        #         device=torch.device(args.device),
        #         tokens_per_batch=args.tokens_per_batch,
        #         num_proc=args.num_proc,
        #         observation_window=args.observation_window,
        #         min_subjects_per_batch=args.min_subjects_per_batch,
        #         use_linear_interpolation=use_linear,
        #     )

        # if "prediction_times" not in features:
        #     pred_times = _build_feature_times_in_return_order(features, label_df)
        #     features["prediction_times"] = pred_times

        # if "target_values" not in features and "target_value" in label_df.columns:
        #     ldf = label_df.sort_values(["subject_id", "prediction_time"])
        #     queues: dict[int, list[float]] = {}
        #     for sid, group in ldf.groupby("subject_id", sort=True):
        #         queues[int(sid)] = list(group["target_value"].astype(float).tolist())
        #     tvals: list[float] = []
        #     for sid in features["subject_ids"]:
        #         sid_int = int(sid)
        #         q = queues.get(sid_int, [])
        #         tvals.append(q.pop(0) if q else np.nan)
        #     features["target_values"] = np.array(tvals, dtype=float)

        with open(feature_output_path, "wb") as f:
            pickle.dump(features, f)

        # with open(training_metrics_file, "w") as output_file:
        #     training_metrics = {
        #         "duration_in_seconds": (datetime.datetime.now() - start_time).total_seconds(),
        #         "total_flops": getattr(total_flops, "total_flops", None),
        #     }
        #     json.dump(training_metrics, output_file)


if __name__ == "__main__":
    main()

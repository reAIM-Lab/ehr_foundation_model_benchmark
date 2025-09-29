import os
import glob
import json
import datetime
from typing import Optional

import femr.transforms
import meds_reader
import femr.models.transformer
import pandas as pd
import pickle
import meds
import pathlib
import torch
from .generate_labels import create_omop_meds_tutorial_arg_parser, LABEL_NAMES


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
    args.add_argument(
        "--num_proc",
        dest="num_proc",
        type=int,
        default=6,
        help="Number of processes to use",
    )
    args.add_argument(
        "--tokens_per_batch",
        dest="tokens_per_batch",
        type=int,
        default=32 * 1024,
        help="The number of tokens per batch to use",
    )
    args.add_argument(
        "--cohort_dir",
        dest="cohort_dir",
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


def read_recursive_parquet(root_dir):
    all_files = glob.glob(os.path.join(root_dir, '**', '*.parquet'), recursive=True)
    df = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)
    return df


def get_motor_features_name(label_name: str, observation_window: Optional[int] = None) -> str:
    if observation_window:
        return label_name + '_motor_' + str(observation_window)
    return label_name + '_motor'


def main():
    args = create_arg_parser().parse_args()
    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=6) as database:
        pretraining_data = pathlib.Path(args.pretraining_data)
        ontology_path = pretraining_data / 'ontology.pkl'

        features_path = pretraining_data / "features"
        features_path.mkdir(exist_ok=True, parents=True)
        flops_path = pretraining_data / "flops"
        flops_path.mkdir(exist_ok=True, parents=True)

        with open(ontology_path, 'rb') as f:
            ontology = pickle.load(f)

        labels = LABEL_NAMES
        if args.cohort_dir is not None:
            if os.path.isdir(args.cohort_dir):
                label_name = os.path.basename(os.path.normpath(args.cohort_dir))
                cohort = read_recursive_parquet(args.cohort_dir)
            else:
                label_name = os.path.basename(os.path.splitext(args.cohort_dir)[0])
                file_extension = os.path.splitext(args.cohort_dir)[1]
                if file_extension.lower() == ".parquet":
                    cohort = pd.read_parquet(args.cohort_dir)
                elif file_extension.lower() == ".csv":
                    cohort = pd.read_csv(args.cohort_dir)
                else:
                    raise RuntimeError(f"Unknown file extension: {file_extension}")
            # We need to cast prediction_time to datetime
            if len(cohort) > 0 and isinstance(cohort.prediction_time.iloc[0], datetime.date):
                cohort["prediction_time"] = pd.to_datetime(cohort["prediction_time"])

            cohort.to_parquet(
                pretraining_data / "labels" / (label_name + '.parquet')
            )
            labels = [label_name]

        for label_name in labels:
            motor_features_name = get_motor_features_name(label_name, args.observation_window)
            feature_output_path = features_path / f"{motor_features_name}.pkl"
            training_metrics_file = flops_path / f"{motor_features_name}.json"
            if feature_output_path.exists():
                print(
                    f"The features for {label_name} already exist at {feature_output_path}, it will be skipped!"
                )
                continue
            labels = pd.read_parquet(
                pretraining_data / "labels" / (label_name + '.parquet')
            )
            typed_labels = [
                meds.Label(
                    subject_id=label["subject_id"],
                    prediction_time=label["prediction_time"],
                    boolean_value=label["boolean_value"],
                )
                for label in labels.to_dict(orient="records")
            ]
            # total_flops = femr.models.transformer.TotalFlops()
            start_time: datetime.datetime = datetime.datetime.now()
            features = femr.models.transformer.compute_features(
                db=database,
                model_path=str(pretraining_data / "motor_model"),
                labels=typed_labels,
                ontology=ontology,
                device=torch.device('cuda'),
                tokens_per_batch=args.tokens_per_batch,
                num_proc=args.num_proc,
                observation_window=args.observation_window,
                # total_flops=total_flops
            )
            with open(feature_output_path, 'wb') as f:
                pickle.dump(features, f)

            # Save the training metrics to the output file
            with open(training_metrics_file, "w") as output_file:
                training_metrics = {
                    "duration_in_seconds": (datetime.datetime.now() - start_time).total_seconds(),
                    # "total_flops": total_flops.total_flops,
                }
                json.dump(training_metrics, output_file)

if __name__ == "__main__":
    main()
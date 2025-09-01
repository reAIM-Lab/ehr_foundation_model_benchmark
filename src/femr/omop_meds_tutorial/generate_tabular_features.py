"""
FEMR also supports generating tabular feature representations, an important baseline for EHR modeling
"""
import os
import glob
import pathlib
import datetime
from typing import Optional

import meds_reader
import pandas as pd
import femr.featurizers
import pickle
from .motor_evaluation.generate_labels import LABEL_NAMES, create_omop_meds_tutorial_arg_parser


def create_arg_parser():
    args = create_omop_meds_tutorial_arg_parser()
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

def get_baseline_features_name(label_name: str, observation_window: Optional[int] = None) -> str:
    if observation_window:
        return label_name + '_' + str(observation_window) + '.pkl'
    return label_name + '.pkl'

def get_baseline_featurizer_name(label_name: str, observation_window: Optional[int] = None) -> str:
    if observation_window:
        return label_name + '_' + str(observation_window) + '_featurizer.pkl'
    return label_name + '_featurizer.pkl'


def main():
    args = create_arg_parser().parse_args()
    pretraining_data = pathlib.Path(args.pretraining_data)
    features_path = pretraining_data / "features"
    features_path.mkdir(exist_ok=True, parents=True)
    label_path = pretraining_data / "labels"
    label_path.mkdir(exist_ok=True, parents=True)
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
            label_path / (label_name + '.parquet')
        )
        labels = [label_name]

    with meds_reader.SubjectDatabase(args.meds_reader, num_threads=32) as database:
        for label_name in labels:
            feature_output_path = features_path / get_baseline_features_name(label_name, args.observation_window)
            if feature_output_path.exists():
                print(
                    f"The features for {label_name} already exist at {feature_output_path}, it will be skipped!")
                continue
            labels = pd.read_parquet(
                label_path / (label_name + '.parquet')
            )
            featurizer = femr.featurizers.FeaturizerList([
                femr.featurizers.AgeFeaturizer(is_normalize=True),
                femr.featurizers.CountFeaturizer(observation_window=args.observation_window),
            ])

            print("Preprocessing")

            featurizer.preprocess_featurizers(database, labels)

            print("Done preprossing, about to featurize")

            with open(features_path / get_baseline_featurizer_name(label_name, args.observation_window), 'wb') as f:
                pickle.dump(featurizer, f)

            features = featurizer.featurize(database, labels)

            print("Done featurizing")

            with open(feature_output_path, 'wb') as f:
                pickle.dump(features, f)


if __name__ == "__main__":
    main()

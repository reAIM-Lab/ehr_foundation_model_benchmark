from pathlib import Path
import datetime
import polars as pl
from meds import train_split, tuning_split, held_out_split
from .subsample_task import get_data_split


def filter_cohort_split(
        cohort_split: pl.DataFrame,
        meds_data: pl.DataFrame,
        observation_window: int
) -> pl.DataFrame:
    if observation_window > 0:
        observation_window_expr = pl.col("time").is_between(
            pl.col("prediction_time") - datetime.timedelta(days=observation_window),
            pl.col("prediction_time")
        )
    else:
        observation_window_expr = pl.col("time") <= pl.col("prediction_time")
    filtered_cohort_split = cohort_split.join(
        meds_data,
        on="subject_id",
    ).filter(observation_window_expr).select(
        "subject_id", "prediction_time", "boolean_value"
    ).unique()
    return filtered_cohort_split


def main(args):
    cohort_dir = Path(args.cohort_dir)
    folder_tasks = [entry.name for entry in cohort_dir.iterdir() if entry.is_dir()]
    file_tasks = [entry.name for entry in cohort_dir.iterdir() if entry.is_file()]
    print(f"{len(folder_tasks) + len(file_tasks)} tasks identified in {args.cohort_dir}.")
    print(f"Tasks: {folder_tasks + file_tasks}")

    meds_dir = Path(args.meds_dir)
    subject_splits_path = meds_dir / "metadata" / "subject_splits.parquet"
    meds_data_parquet_files = (meds_dir / "data").rglob("*.parquet")
    print(f"Loading subject_splits.parquet from {subject_splits_path}")
    subject_splits = pl.read_parquet(subject_splits_path)
    print(f"Loading the MEDS data")
    meds_data = pl.read_parquet(
        list(meds_data_parquet_files),
        columns=["subject_id", "time", "table"]
    ).filter(
        pl.col("table").is_in(["condition", "drug_exposure"])
    )
    output_dir = Path(args.output_dir)
    for task in folder_tasks + file_tasks:
        print(f"\nStart processing: {task}")
        task_path = cohort_dir / task
        if task_path.is_file():
            if task_path.suffix != ".parquet":
                print(f"{task_path} is not a valid parquet file, therefore skip")
                continue
            cohort_data = pl.read_parquet(task_path)
            task_name = task_path.stem
        elif task_path.is_dir():
            cohort_data = pl.read_parquet(list(task_path.rglob('*.parquet')))
            task_name = task
        else:
            print(f"{task_path} is neither a valid parquet file and nor a folder, therefore skip")
            continue

        output_task_dir = output_dir / task_name
        output_task_dir.mkdir(exist_ok=True)

        for split in [train_split, tuning_split, held_out_split]:
            cohort_split = get_data_split(cohort_data, subject_splits, split)
            print(f"{task_name} {split} before filter: {len(cohort_split)}")
            filtered_cohort_split = filter_cohort_split(
                cohort_split=cohort_split,
                meds_data=meds_data,
                observation_window=args.observation_window
            )
            print(f"{task_name} {split} after filter: {len(filtered_cohort_split)}")
            filtered_cohort_split.write_parquet(output_task_dir / f"{split}.parquet")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for applying the feature extraction criterion, "
                    "which is that patients need to have at least 1 condition or "
                    "procedure in the specified observation window"
    )
    parser.add_argument(
        "--cohort_dir",
        dest="cohort_dir",
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
        "--observation_window",
        dest="observation_window",
        action="store",
        type=int,
        default=0,
        required=False,
    )
    main(
        parser.parse_args()
    )

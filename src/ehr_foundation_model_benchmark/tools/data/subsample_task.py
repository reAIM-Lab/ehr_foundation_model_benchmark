import argparse
from pathlib import Path
import polars as pl

DEFAULT_TASKS = [
    'AMI', 'Celiac', 'CLL', 'HTN', 'Ischemic_Stroke', 'MASLD', 'Osteoporosis', 'Pancreatic_Cancer', 'SLE', 'T2DM'
]


def sample(df, n_max, min_obs=None, label_col="boolean_value"):
    """
    Stratified sampling function that randomly samples `n_max` rows from the input DataFrame `df`,
    preserving the class distribution of the 'boolean_value' column as much as possible.

    If the exact stratified proportions do not sum to `n_max` due to rounding, the function adjusts 
    by assigning any remaining rows to the minority class.

    Parameters:
    -----------
    df : pl.DataFrame
        A Polars DataFrame containing the label column for stratification.
    n_max : int
        The maximum total number of rows to sample.
    min_obs : float or None
        Minimum prevalence of samples required for the minority class. If none, this will just use original data prevalences.
    label_col : str
        The name of the column used for stratification.

    Returns:
    --------
    df_sampled : pl.DataFrame
        A new DataFrame of length `n_max`, stratified by label_col.
    """

    # Project timings and come up with 
    df_counts = df.group_by(label_col).len()
    total_n = df_counts["len"].sum()

    df_counts = df_counts.with_columns(
        ((df_counts["len"] / df_counts["len"].sum())).alias("prevalence")
    )
    df_counts = df_counts.with_columns(
        ((df_counts["len"] / df_counts["len"].sum()) * n_max).cast(pl.Int64()).alias("sample_size")
    ).drop('len').drop('prevalence')

    # Enforce minimum prevalence for the minority class and compute required sample sizes
    if min_obs is not None:
        min_row = df_counts.sort("sample_size").row(0)
        min_class = min_row[0]
        min_required = int(min_obs * n_max)

        available_min_class = df.filter(pl.col(label_col) == min_class).shape[0]

        if available_min_class < min_required:
            # Adjust n_max based on available data to satisfy min_obs prevalence
            n_max = int(available_min_class / min_obs)
            min_required = available_min_class  # We'll use all available minority samples

            # Recompute sample sizes
            df_counts = df_counts.with_columns(
                pl.when(pl.col(label_col) == min_class)
                .then(pl.lit(available_min_class))
                .otherwise(pl.lit(n_max - available_min_class))
                .alias("sample_size")
            )
        else:
            df_counts = df_counts.with_columns(
                pl.when(pl.col(label_col) == min_class)
                .then(pl.lit(min_required))
                .otherwise(pl.lit(n_max - min_required))
                .alias("sample_size")
            )

    # Adjust for rounding
    missing_samples = n_max - df_counts["sample_size"].sum()

    if missing_samples > 0:
        min_class = df_counts.sort("sample_size").select(label_col)[0]
        df_counts = df_counts.with_columns(
            pl.when(pl.col(label_col) == min_class)
            .then(pl.col("sample_size") + missing_samples)
            .otherwise(pl.col("sample_size"))
            .alias("sample_size")
        )

    print(df_counts)

    # Based on computed sample sizes, sample from original df
    sampled_dfs = []
    for class_label, class_count in df_counts.iter_rows():
        class_subset = df.filter(pl.col(label_col) == class_label)
        sampled_df = class_subset.sample(n=class_count, shuffle=True, with_replacement=False)
        sampled_dfs.append(sampled_df)

    df_sampled = pl.concat(sampled_dfs)
    return df_sampled, total_n


def get_data_split(cohort: pl.DataFrame, subject_splits: pl.DataFrame, split: str) -> pl.DataFrame:
    assert split in ["train", "tuning", "held_out"]
    split_data = cohort.join(
        subject_splits.filter(pl.col("split") == split).select("subject_id"),
        on='subject_id'
    )
    return split_data


def main(args):
    cohort_dir = Path(args.cohort_dir)
    folder_tasks = [entry.name for entry in cohort_dir.iterdir() if entry.is_dir()]
    file_tasks = [entry.name for entry in cohort_dir.iterdir() if entry.is_file()]
    print(f"{len(folder_tasks) + len(file_tasks)} tasks identified in {args.cohort_dir}.")
    print(f"Tasks: {folder_tasks + file_tasks}")

    n_train = 80000
    n_tune = 20000
    n_test = 50000

    meds_dir = Path(args.meds_dir)
    subject_splits_path = meds_dir / "metadata" / "subject_splits.parquet"
    print(f"Loading subject_splits.parquet from {subject_splits_path}")
    subject_splits = pl.read_parquet(subject_splits_path)
    output_dir = Path(args.output_dir)

    for task in folder_tasks + file_tasks:
        print(f"Start processing: {task}")
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

        df_train = get_data_split(cohort_data, subject_splits, "train")
        df_train, train_count = sample(df_train, n_train)
        df_train.write_parquet(output_task_dir / "train.parquet")

        df_tune = get_data_split(cohort_data, subject_splits, "tuning")
        df_tune, val_count = sample(df_tune, n_tune)
        df_tune.write_parquet(output_task_dir / "tuning.parquet")

        df_test = get_data_split(cohort_data, subject_splits, "held_out")
        df_test, test_count = sample(df_test, n_test)
        df_test.write_parquet(output_task_dir / "held_out.parquet")

        print(f"Sampled cohort size for {task}: train - {train_count}, tuning - {val_count}, held_out - {test_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments"
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
    main(
        parser.parse_args()
    )

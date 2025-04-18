import argparse
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from utils import count_events

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

def main(args):

    base_path = Path(args.input_meds)
    
    tasks = ['AMI', 'Celiac', 'CLL', 'HTN', 'Ischemic_Stroke', 'MASLD', 'Osteoporosis', 'Pancreatic_Cancer', 'SLE', 'T2DM']

    n_train = 80000
    n_tune = 20000
    n_test = 50000

    total = 0

    total_lengths = []

    for task in tasks:
        train_labels_path = base_path / f"task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years/{task}/train.parquet"
        tune_labels_path = base_path / f"task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years/{task}/tuning.parquet"
        test_labels_path = base_path / f"task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years/{task}/held_out.parquet"

        new_path = base_path / f"task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years_sample/{task}"
        new_path.mkdir(parents=True, exist_ok=True)

        train_path = base_path / "post_transform/data/train"
        tune_path = base_path / "post_transform/data/tuning"
        test_path = base_path / "post_transform/data/held_out"

        train_files = sorted(train_path.glob("*.parquet"))
        tune_files = sorted(tune_path.glob("*.parquet"))
        test_files = sorted(test_path.glob("*.parquet"))

        df_train = pl.read_parquet(train_labels_path)
        df_train, train_count = sample(df_train, n_train)
        duplicates = df_train.filter(df_train.is_duplicated())
        df_train.write_parquet(new_path / "train.parquet")

        # train_events = count_events(df_train, train_files)
        # lengths = [len(lst) for lst in train_events]
        # lengths = [min(x, 5000) for x in lengths]

        # plt.figure(figsize=(6, 4))
        # plt.hist(lengths, bins=50)
        # plt.xlabel("Number of Events")
        # plt.ylabel("Frequency")
        # plt.title("Distribution of Event Count")
        # plt.grid(True)
        # plt.tight_layout()

        # plt.savefig(f"plots/events_per_sample_{task}.pdf")

        df_tune = pl.read_parquet(tune_labels_path)
        df_tune, val_count = sample(df_tune, n_tune)
        df_tune.write_parquet(new_path / "tuning.parquet")

        df_test = pl.read_parquet(test_labels_path)
        df_test, test_count = sample(df_test, n_test)
        df_test.write_parquet(new_path / "held_out.parquet")

        total += test_count + train_count + val_count

    print(total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments"
    )
    parser.add_argument(
        "--input_meds",
        dest="input_meds",
        action="store",
        default="/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/"
    )

    main(
        parser.parse_args()
    )
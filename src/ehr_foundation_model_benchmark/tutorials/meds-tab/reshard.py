import argparse
from pathlib import Path
import polars as pl


def main(args):
    print("Reading cohort")
    cohort = pl.read_parquet(args.cohort_input)
    if args.ratio < 1.0:
        cohort = cohort.sample(fraction=args.ratio, seed=42)  # Sample the cohort if ratio < 1.0
    if args.split == "all":
        splits = ["train", "tuning", "held_out"]
    else:
        splits = [args.split]
    for split in splits:
        print("Reading split", split)
        output_folder = Path(args.cohort_output) / split
        if not output_folder.exists():
            output_folder.mkdir(parents=True)
        for shard_path in (Path(args.meds_data) / split).glob("*parquet"):
            print("Reading shard", shard_path.stem)
            output_shard_path = output_folder / shard_path.name
            output_shard = cohort.filter(pl.col("subject_id").is_in(pl.read_parquet(shard_path).select("subject_id")))
            output_shard = output_shard.with_columns(
                pl.col("prediction_time").cast(pl.Datetime("us"))
            )
            output_shard.write_parquet(output_shard_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for preparing Motor")
    parser.add_argument(
        "--meds_data",
        dest="meds_data",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--cohort_input",
        dest="cohort_input",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--cohort_output",
        dest="cohort_output",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--split",
        dest="split",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--ratio",
        dest="ratio",
        action="store",
        default=1.0,
        type=float,
        help="Ratio of the cohort to use for training. Default is 1.0 (use all data).",
    )
    main(parser.parse_args())


# python reshard.py --cohort_input /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/motor/labels/hf_readmission_meds.parquet  --meds_data /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform/data --cohort_output /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/labels/hf_readmission_meds

# python reshard.py --cohort_input /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years/AMI/train.parquet  --meds_data /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform/data --cohort_output /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/labels/ami --split train

# python reshard.py --cohort_input /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years/AMI/held_out.parquet  --meds_data /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform/data --cohort_output /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/labels/ami --split held_out
"""
Script to shard labels as the data is sharded
No need to call it manually, it is called by other scripts
"""

import argparse
from pathlib import Path
import polars as pl


def main(args):
    print("Reading cohort")
    cohort = pl.read_parquet(args.cohort_input)
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
            # output_shard = cohort.filter(pl.col("subject_id").is_in(pl.read_parquet(shard_path)["subject_id"]))
            shard_subjects = pl.read_parquet(shard_path).select("subject_id")
            output_shard = cohort.join(shard_subjects, on="subject_id", how="inner")
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
    main(parser.parse_args())

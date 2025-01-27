import os
import argparse
import shutil
from pathlib import Path
import polars as pl


def main(args):
    # Create output directories for metadata and data
    sample_metadata_path = Path(args.output_meds) / "metadata"
    sample_metadata_path.mkdir(exist_ok=True)
    sample_data_path = Path(args.output_meds) / "data"
    sample_data_path.mkdir(exist_ok=True)

    # Copy the 'codes.parquet' file from input to output metadata folder
    shutil.copyfile(Path(args.input_meds) / "metadata" / "codes.parquet", sample_metadata_path / "codes.parquet")

    # Read the 'subject_splits' file and create a sample of 5000 subjects
    subject_splits = pl.read_parquet(os.path.join(args.input_meds, "metadata/subject_splits.parquet"))
    subject_splits_sample = subject_splits.sample(args.sample_size, seed=42)
    subject_splits_sample.write_parquet(sample_metadata_path / "subject_splits.parquet")

    # Process and filter each parquet file in the data folder
    for parquet_file_path in (Path(args.input_meds) / "data").glob("*.parquet"):
        source_data = pl.read_parquet(parquet_file_path)
        sample_source_data = source_data.filter(source_data["subject_id"].is_in(subject_splits_sample["subject_id"]))
        sample_source_data = sample_source_data.with_columns(
            pl.coalesce(pl.col("unit"), pl.lit("")).alias("unit")
        )
        sample_source_data.write_parquet(sample_data_path / parquet_file_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for sampling Columbia MEDS data"
    )
    parser.add_argument(
        "--input_meds",
        dest="input_meds",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--output_meds",
        dest="output_meds",
        action="store",
        required=True,
    )
    parser.add_argument(
        "--sample_size",
        dest="sample_size",
        action="store",
        type=int,
        default=5000,
        required=False,
    )
    main(
        parser.parse_args()
    )
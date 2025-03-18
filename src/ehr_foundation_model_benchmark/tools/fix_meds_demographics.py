"""
MEDS Demographics Processor

This script processes Medical Event Data Set (MEDS) files to ensure consistent demographic 
information across datasets. It enriches MEDS data with standardized demographic codes from 
OMOP person tables and concept tables.

The script performs the following operations:
1. Loads OMOP person and concept tables
2. Creates a mapping between concept IDs and standardized codes
3. Processes MEDS files across train/tuning/held-out splits
4. For each file:
   - Identifies birth events
   - Removes existing demographics
   - Adds standardized gender, race, and ethnicity codes
   - Merges and sorts all events

Usage:
    python fix_meds_demographics.py --omop_person_dir /path/to/person \
                               --concept_dir /path/to/concept \
                               --meds_dir /path/to/meds \
                               --output_dir /path/to/output

Requirements:
    - polars
    - tqdm
    - meds module with train_split, tuning_split, held_out_split, and birth_code
"""

import argparse
import os
import glob
from pathlib import Path
from meds import train_split, tuning_split, held_out_split, birth_code
import polars as pl
from tqdm import tqdm


def main(args):
    """
    Main function to process MEDS files with standardized demographics.

    Args:
        args (argparse.Namespace): Command line arguments containing:
            - omop_person_dir: Directory containing OMOP person tables
            - concept_dir: Directory containing OMOP concept tables
            - meds_dir: Directory containing MEDS data files
            - output_dir: Directory to save processed MEDS files
    """
    # Create output directory
    meds_data_dir = os.path.join(args.meds_dir, "data")
    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(exist_ok=True, parents=True)

    # Load person and concept tables
    person = pl.read_parquet(
        glob.glob(os.path.join(args.omop_person_dir, '**', '*.parquet'), recursive=True)
    )
    concept = pl.read_parquet(
        glob.glob(os.path.join(args.concept_dir, '**', '*.parquet'), recursive=True)
    )

    # Create concept mapping for demographics
    concept_mapping_df = pl.concat([
        person.select(pl.col("gender_concept_id").alias("concept_id")),
        person.select(pl.col("race_concept_id").alias("concept_id")),
        person.select(pl.col("ethnicity_concept_id").alias("concept_id"))
    ]).unique().join(
        concept, on="concept_id"
    ).with_columns(
        pl.concat_str(pl.col("vocabulary_id"), pl.lit("/"), pl.col("concept_code")).alias("code")
    ).filter(
        # Filter out concept IDs that represent unknown/missing values
        ~pl.col("concept_id").is_in([0, 44814649, 44814653])
    ).select(
        "concept_id",
        "code"
    )

    # Convert concept mapping to dictionary for efficient lookups
    concept_code_dict = concept_mapping_df.to_dict(as_series=False)
    concept_code_dict = dict(zip(concept_code_dict["concept_id"], concept_code_dict["code"]))

    # Process each data split (train, tuning, held-out)
    for split in [train_split, tuning_split, held_out_split]:
        print(f"Processing {split}")
        parquet_files = glob.glob(os.path.join(meds_data_dir, split, '*.parquet'), recursive=True)
        output_data_split_dir = output_data_dir / split
        output_data_split_dir.mkdir(exist_ok=True, parents=True)
        # Process each file in the split
        for parquet_file in tqdm(parquet_files, total=len(parquet_files)):
            output_file = output_data_split_dir / os.path.basename(parquet_file)

            # Read MEDS file
            meds = pl.read_parquet(parquet_file)

            # Extract birth events (used for demographic timing)
            meds_birth_datetime = meds.filter(pl.col("code") == birth_code)

            # Remove existing demographic data
            meds_without_demographics = meds.filter(pl.col("table") != "person")

            # Join person data with birth events to get timing information
            partition_person = person.join(
                meds_birth_datetime.select("subject_id", "time", "table"),
                left_on="person_id",
                right_on="subject_id",
            )

            # Create standardized gender events
            gender_events = partition_person.select(
                pl.col("person_id").alias("subject_id"),
                "time",
                pl.col("gender_concept_id").replace_strict(concept_code_dict, default="Gender/Unknown").alias("code"),
            )

            # Create standardized race events
            race_events = partition_person.select(
                pl.col("person_id").alias("subject_id"),
                "time",
                pl.col("race_concept_id").replace_strict(concept_code_dict, default="Race/Unknown").alias("code"),
            )

            # Create standardized ethnicity events
            ethnicity_events = partition_person.select(
                pl.col("person_id").alias("subject_id"),
                "time",
                pl.col("ethnicity_concept_id").replace_strict(concept_code_dict, default="Ethnicity/Unknown").alias(
                    "code"),
            )

            # Combine demographic events and add required null columns
            new_demographics_events = pl.concat([
                gender_events, race_events, ethnicity_events
            ]).with_columns(
                pl.lit(None, dtype=pl.Float32).alias("numeric_value"),
                pl.lit(None, dtype=pl.Datetime).alias("end"),
                pl.lit("person").alias("table"),
                pl.lit(None, dtype=pl.String).alias("text_value"),
                pl.lit(None, dtype=pl.String).alias("unit"),
                pl.lit(None, dtype=pl.Int32).alias("visit_id"),
            )

            # Combine birth events, new demographics, and existing non-demographic events
            new_meds = pl.concat([
                meds_birth_datetime,
                new_demographics_events,
                meds_without_demographics
            ]).sort(["subject_id", "time", "code"])

            assert (
                len(new_meds) >= len(meds),
                f"the patched meds partition at {output_file} "
                f"must have more rows than the original partition at {parquet_file}"
            )
            # Write processed file
            new_meds.write_parquet(output_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Fix MEDS demographics by adding standardized codes")
    arg_parser.add_argument(
        "--omop_person_dir",
        dest="omop_person_dir",
        required=True,
        help="Directory containing OMOP person tables (parquet format)"
    )
    arg_parser.add_argument(
        "--concept_dir",
        dest="concept_dir",
        required=True,
        help="Directory containing OMOP concept tables (parquet format)"
    )
    arg_parser.add_argument(
        "--meds_dir",
        dest="meds_dir",
        required=True,
        help="Directory containing MEDS data files"
    )
    arg_parser.add_argument(
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Directory to save processed MEDS files"
    )
    main(arg_parser.parse_args())

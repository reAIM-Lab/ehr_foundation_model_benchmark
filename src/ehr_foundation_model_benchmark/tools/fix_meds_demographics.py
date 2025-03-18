import argparse
import os
import glob
from pathlib import Path
from meds import train_split, tuning_split, held_out_split, birth_code
import polars as pl
from tqdm import tqdm


def main(args):
    # Load the person table first
    meds_data_dir = os.path.join(args.meds_dir, "data")
    output_data_dir = Path(args.output_dir) / "data"
    output_data_dir.mkdir(exist_ok=True, parents=True)
    person = pl.read_parquet(
        glob.glob(os.path.join(args.omop_person_dir, '**', '*.parquet'), recursive=True)
    )
    concept = pl.read_parquet(
        glob.glob(os.path.join(args.concept_dir, '**', '*.parquet'), recursive=True)
    )
    concept_mapping_df = pl.concat([
        person.select(pl.col("gender_concept_id").alias("concept_id")),
        person.select(pl.col("race_concept_id").alias("concept_id")),
        person.select(pl.col("ethnicity_concept_id").alias("concept_id"))
    ]).unique().join(
        concept, on="concept_id"
    ).with_columns(
        pl.concat_str(pl.col("vocabulary_id"), pl.lit("/"), pl.col("concept_code")).alias("code")
    ).filter(
        ~pl.col("concept_id").is_in([0, 44814649, 44814653])
    ).select(
        "concept_id",
        "code"
    )
    # Then convert to dictionary
    concept_code_dict = concept_mapping_df.to_dict(as_series=False)
    concept_code_dict = dict(zip(concept_code_dict["concept_id"], concept_code_dict["code"]))

    for split in [train_split, tuning_split, held_out_split]:
        print(f"Processing {split}")
        parquet_files = glob.glob(os.path.join(meds_data_dir, split, '*.parquet'), recursive=True)
        for parquet_file in tqdm(parquet_files, total=len(parquet_files)):
            output_file = output_data_dir / os.path.basename(parquet_file)
            meds = pl.read_parquet(parquet_file)
            meds_birth_datetime = meds.filter(pl.col("code") == birth_code)
            meds_without_demographics = meds.filter(pl.col("table") != "person")
            partition_person = person.join(
                meds_birth_datetime.select("subject_id", "time", "table"),
                left_on="person_id",
                right_on="subject_id",
            )
            gender_events = partition_person.select(
                pl.col("person_id").alias("subject_id"),
                "time",
                pl.col("gender_concept_id").replace_strict(concept_code_dict, default="Gender/Unknown").alias("code"),
            )
            race_events = partition_person.select(
                pl.col("person_id").alias("subject_id"),
                "time",
                pl.col("race_concept_id").replace_strict(concept_code_dict, default="Race/Unknown").alias("code"),
            )
            ethnicity_events = partition_person.select(
                pl.col("person_id").alias("subject_id"),
                "time",
                pl.col("ethnicity_concept_id").replace_strict(concept_code_dict, default="Ethnicity/Unknown").alias(
                    "code"),
            )
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
            new_meds = pl.concat([
                meds_birth_datetime,
                new_demographics_events,
                meds_without_demographics
            ]).sort(["subject_id", "time", "code"])
            new_meds.write_parquet(output_file)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser("Fix MEDS demographics")
    arg_parser.add_argument(
        "--omop_person_dir",
        dest="omop_person_dir",
        required=True
    )
    arg_parser.add_argument(
        "--concept_dir",
        dest="concept_dir",
        required=True
    )
    arg_parser.add_argument(
        "--meds_dir",
        dest="meds_dir",
        required=True
    )
    arg_parser.add_argument(
        "--output_meds_dir",
        dest="output_meds_dir",
        required=True
    )
    main(arg_parser.parse_args())

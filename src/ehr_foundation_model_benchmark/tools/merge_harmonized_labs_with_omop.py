import argparse
from pathlib import Path
import polars as pl
import shutil


def main(args):
    # Use Path for all path manipulations
    omop_folder_path = Path(args.omop_folder)
    output_folder_path = Path(args.output_folder)
    harmonized_labs_folder_path = Path(args.harmonized_labs_folder)

    # Create the output folder if it does not exist
    output_folder_path.mkdir(parents=True, exist_ok=True)

    # Copy everything from the omop folder to the new output folder, excluding the 'measurement' directory
    for item in omop_folder_path.iterdir():
        if item.name != 'measurement':
            if item.is_dir():
                shutil.copytree(item, output_folder_path / item.name)
            else:
                shutil.copy2(item, output_folder_path / item.name)

    # Prepare the measurement folder in the destination
    measurement_folder_path = output_folder_path / "measurement"
    measurement_folder_path.mkdir(exist_ok=True)

    # Process each Parquet file without assuming 'v5' in the file names
    for parquet_file in harmonized_labs_folder_path.glob("*.parquet"):
        labs = pl.scan_parquet(parquet_file)
        # Lazily rename columns
        labs = labs.with_columns([
            pl.col("value_as_number").alias("original_value_as_number"),
            pl.col("harmonized_value_as_number").alias("value_as_number"),
            pl.col("unit_concept_id").alias("original_unit_concept_id"),
            pl.col("harmonized_unit_concept_id").alias("unit_concept_id")
        ])
        # Write output using lazy execution
        labs.write_parquet(measurement_folder_path / parquet_file.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge harmonized labs with omop")
    parser.add_argument("omop_folder", help="Path to the OMOP folder")
    parser.add_argument("harmonized_labs_folder", help="Path to the harmonized labs folder")
    parser.add_argument("output_folder", help="Path to the output folder")
    args = parser.parse_args()
    main(args)

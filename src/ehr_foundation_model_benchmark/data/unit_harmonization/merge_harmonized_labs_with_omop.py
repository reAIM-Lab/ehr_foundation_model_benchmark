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
            print(f"Copying {item.name} from {omop_folder_path} to {output_folder_path}")
            if item.is_dir():
                shutil.copytree(item, output_folder_path / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, output_folder_path / item.name)
        else:
            print(f"Skipping copying measurement")

    # Prepare the measurement folder in the destination
    measurement_folder_path = output_folder_path / "measurement"
    measurement_folder_path.mkdir(exist_ok=True)

    # Process each Parquet file without assuming 'v5' in the file names
    for parquet_file in harmonized_labs_folder_path.glob("*v5*parquet"):
        print(f"Converting {parquet_file.name} from {harmonized_labs_folder_path} to {measurement_folder_path}")
        labs = pl.scan_parquet(parquet_file)
        # Lazily rename columns
        labs = labs.with_columns([
            pl.col("value_as_number").alias("original_value_as_number"),
            pl.col("harmonized_value_as_number").alias("value_as_number"),
            pl.col("unit_concept_id").alias("original_unit_concept_id"),
            pl.col("harmonized_unit_concept_id").alias("unit_concept_id")
        ]).with_columns([
            pl.col("visit_occurrence_id").cast(pl.Int32),
            pl.col("visit_detail_id").cast(pl.Int32),
        ])
        # Write output using lazy execution
        labs.collect().write_parquet(measurement_folder_path / parquet_file.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge harmonized labs with omop")
    parser.add_argument(
        "--omop_folder",
        dest="omop_folder",
        help="Path to the OMOP folder",
        required=True,
    )
    parser.add_argument(
        "--harmonized_labs_folder",
        dest="harmonized_labs_folder",
        help="Path to the harmonized labs folder",
        required = True,
    )
    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        help="Path to the output folder",
        required=True,
    )
    args = parser.parse_args()
    main(args)

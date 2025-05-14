"""
Unit Harmonization Analysis Script

This script processes harmonized measurement files to:
1. Compute majority units for each measurement concept
2. Classify unit and value matching types
3. Generate comprehensive statistics on unit and value harmonization

Usage:
    python unit_harmonization_analysis.py
        --harmonized_measurement_dir /path/to/harmonized/measurements
"""

import glob
import os
from argparse import ArgumentParser
import pandas as pd

from ehr_foundation_model_benchmark.data.unit_harmonization.mappings import (
    compute_most_common_units,
)

def process_harmonized_file(file: str) -> None:
    """
    Process a single harmonized measurement file.

    Args:
        file (str): Path to the harmonized parquet file
    """
    try:
        print()
        print("File", file)

        # Read parquet file
        data = pd.read_parquet(file)

        # Compute most common units
        most_common_units = compute_most_common_units()

        # Convert most common units to a dictionary
        most_common_units_df = pd.Series(most_common_units).reset_index()
        most_common_units_df['index'] = most_common_units_df['index'].astype(int)
        most_common_units_df.set_index('index', inplace=True)
        most_common_units = most_common_units_df.to_dict()[0]

        # Get majority units
        print("Getting majority units")
        data['majority_unit_id'] = pd.DataFrame(
            data['measurement_concept_id']
        ).replace(to_replace=most_common_units)['measurement_concept_id']

        print('Finished filling in majority units')

        # Classify unit match types
        data['unit_match_type'] = 'to_fill'
        data['harmonized_unit_concept_id'].fillna(0, inplace=True)

        # Detailed unit match type classification
        data.loc[
            (data['harmonized_unit_concept_id'] == data['majority_unit_id']) &
            (data['majority_unit_id'] == 0),
            'unit_match_type'
        ] = 'Mapped to majority unit (nan)'

        data.loc[
            (data['harmonized_unit_concept_id'] == data['majority_unit_id']) &
            (data['majority_unit_id'] != 0),
            'unit_match_type'
        ] = 'Mapped to majority unit (non-nan)'

        data.loc[
            (data['harmonized_unit_concept_id'] != data['majority_unit_id']) &
            (data['majority_unit_id'] != 0),
            'unit_match_type'
        ] = 'Mapped to minority unit'

        data.loc[
            (data['harmonized_unit_concept_id'] == 0) &
            (data['majority_unit_id'] != 0),
            'unit_match_type'
        ] = 'Unmapped nan'

        # Classify value match types
        data['value_match_type'] = 'to_fill'
        data.loc[
            ~(data['harmonized_value_as_number'].isna()),
            'value_match_type'
        ] = 'Numerical value'

        data.loc[
            (data['harmonized_value_as_number'].isna()) &
            ~((data['value_as_concept_id'].isna())),
            'value_match_type'
        ] = 'Concept value (no number)'

        data.loc[
            (data['harmonized_value_as_number'].isna()) &
            ((data['value_as_concept_id'].isna())),
            'value_match_type'
        ] = 'Nan concept and number'

        # Classify original unit type
        data['original_unit_type'] = 'Non-Nan'
        data.loc[
            ((data['unit_concept_id'].isna()) | (data['unit_concept_id']==0)),
            'original_unit_type'
        ] = 'Nan'

        # Classify original value type
        data['original_value_type'] = 'to_fill'
        data.loc[
            ~(data['value_as_number'].isna()),
            'original_value_type'
        ] = 'Numerical value'

        data.loc[
            (data['value_as_number'].isna()) &
            ~((data['value_as_concept_id'].isna())),
            'original_value_type'
        ] = 'Concept value (no number)'

        data.loc[
            (data['value_as_number'].isna()) &
            ((data['value_as_concept_id'].isna())),
            'original_value_type'
        ] = 'Nan concept and number'

        # Compute counts
        counts = data.groupby(
            by=[
                "unit_match_type",
                "value_match_type",
                "original_unit_type",
                'original_value_type'
            ]
        ).size().reset_index()

        # Save counts to CSV
        output_csv = file.replace(".snappy.parquet", "-v5-processed_unit_value_counts.csv")
        counts.to_csv(output_csv, index=False)
        print(f"Saved counts to {output_csv}")

    except Exception as e:
        print(f"Error processing file {file}: {e}")

def main():
    """
    Main function to parse arguments and process harmonized measurement files.
    """
    argparser = ArgumentParser("Unit Harmonization Analysis")
    argparser.add_argument(
        "--harmonized_measurement_dir",
        dest="harmonized_measurement_dir",
        type=str,
        required=True,
        help="Directory containing harmonized measurement parquet files"
    )
    args = argparser.parse_args()

    # Validate input directory
    if not os.path.isdir(args.harmonized_measurement_dir):
        print(f"Directory not found: {args.harmonized_measurement_dir}")
        return

    # Find parquet files
    files = glob.glob(os.path.join(args.harmonized_measurement_dir, "*.parquet"))

    if not files:
        print(f"No parquet files found in {args.harmonized_measurement_dir}")
        return

    # Process each file
    for file in files:
        process_harmonized_file(file)

if __name__ == "__main__":
    main()
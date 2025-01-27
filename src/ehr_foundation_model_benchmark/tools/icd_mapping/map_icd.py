import argparse
from pathlib import Path
import polars as pl

from utils import try_loading_gem_file, map_stochastically

def main(args):
    sample_metadata_path = Path(args.output_meds) / "metadata"
    data_path = Path(args.output_meds) / "data"
    vocabulary_cache_dir = sample_metadata_path / "vocabulary_cache"

    mapping_path = vocabulary_cache_dir / "mapping"
    mapping_path.mkdir(exist_ok=True)

    translated_path = Path(args.output_meds) / "translated"
    translated_path.mkdir(exist_ok=True)

    if not vocabulary_cache_dir.exists():
        raise FileNotFoundError(
            f"{vocabulary_cache_dir} does not exist, "
            f"the prevalence table must exist in {vocabulary_cache_dir}"
        )
    try:
        wildcard_path = vocabulary_cache_dir / "prevalence" / "*.parquet"
        prevalence = pl.scan_parquet(wildcard_path)
    except pl.exceptions.ComputeError:
        # Handle the error
        raise pl.exceptions.ComputeError(
            f"Error loading the prevalence parquet files from: {vocabulary_cache_dir}"
        )
    
    gem_mapping = try_loading_gem_file(args.target_vocabulary)

    source_to_target_mappings = map_stochastically(gem_mapping, prevalence).collect()
    source_to_target_mappings.write_parquet(mapping_path / "data.parquet")

    #print(gem_mapping.collect())
    #print(prevalence.collect())
    #print(source_to_target_mappings)
    
    # Perform mapping on individual parquet files
    for parquet_file_path in data_path.glob("*.parquet"):
        source_data = pl.read_parquet(parquet_file_path)
        
        #icd_codes = [code for code in source_data['code'].to_list() if code.startswith('ICD')]
        #icd10_pcs = [code for code in source_data['code'].to_list() if code.startswith('ICD10PCS')]
        #icd9_pcs = [code for code in source_data['code'].to_list() if code.startswith('ICD9Proc')]

        if args.source_vocabulary == 'ICD9':
            # Format source ICD9 data zero padding to be consistent with mapping 
            source_data = source_data.with_columns(
                pl.col("code")
                .map_elements(lambda code: (
                    code if "ICD9" not in code else
                    # Add a dot after "/" if it's missing
                    code if "." in code.split("/")[1] else f"{code.split('/')[0]}/{code.split('/')[1][:3]}."
                ), return_dtype=pl.Utf8,)
                .map_elements(lambda code: (
                    code if "ICD9CM/" not in code else
                    # Ensure the character length after "/" is 6 for diagnosis
                    code if len(code.split("/")[1]) == 6 else f"{code.split('/')[0]}/{code.split('/')[1].ljust(6, '0')}"
                ), return_dtype=pl.Utf8,)
                .map_elements(lambda code: (
                    code if "ICD9Proc/" not in code else
                    # Ensure the character length after "/" is 5 for procedures
                    code if len(code.split("/")[1]) == 6 else f"{code.split('/')[0]}/{code.split('/')[1].ljust(5, '0')}"
                ), return_dtype=pl.Utf8,)
                .alias("code")
            )

        updated_source_data = (
            source_data
            .join(source_to_target_mappings, left_on="code", right_on="source_code", how="left")
            .with_columns(
                # Use 'selected_target_code' if it exists, otherwise keep the original 'code'
                pl.when(pl.col("selected_target_code").is_not_null())
                .then(pl.col("selected_target_code"))
                .otherwise(pl.col("code"))
                .alias("code")
            )
            .select(source_data.columns)
            .filter(~pl.col("code").str.starts_with(args.source_vocabulary)) # Drop any codes that were not mapped
        )

        output_file_path = translated_path / parquet_file_path.name
        updated_source_data.write_parquet(output_file_path)

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
        "--target_vocabulary",
        dest="target_vocabulary",
        action="store",
        required=True,
        choices=['ICD9', 'ICD10']
    )
    parser.add_argument(
        "--source_vocabulary",
        dest="source_vocabulary",
        action="store",
        required=True,
        choices=['ICD9', 'ICD10']
    )
    main(
        parser.parse_args()
    )
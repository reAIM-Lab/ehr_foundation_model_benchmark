import argparse
from pathlib import Path
import polars as pl

from collections import Counter

def main(args):
    sample_metadata_path = Path(args.output_meds) / "metadata"
    data_path = Path(args.output_meds) / "data"

    vocabulary_cache_dir = sample_metadata_path / "vocabulary_cache"
    vocabulary_cache_dir.mkdir(exist_ok=True)

    code_counter = Counter()

    for parquet_file_path in data_path.glob("*.parquet"):
        source_data = pl.read_parquet(parquet_file_path)
        icd_codes = [code for code in source_data['code'].to_list() if code.startswith('ICD')]
        code_counter.update(icd_codes)

    total_codes = sum(code_counter.values())

    # Convert the Counter to a Polars DataFrame and compute prevalence
    code_freq_df = pl.DataFrame({
        'concept_code': list(code_counter.keys()),
        'frequency': list(code_counter.values())
    })
    code_freq_df = code_freq_df.with_columns(
        (pl.col("frequency") / total_codes).alias("prevalence")
    )
    code_prevalence_df = code_freq_df.drop("frequency")

    prevalence_path = vocabulary_cache_dir / "prevalence"
    prevalence_path.mkdir(exist_ok=True)

    output_path = prevalence_path / "data.parquet"
    code_prevalence_df.write_parquet(output_path)

    code_df = code_prevalence_df.drop("prevalence")
    code_df = code_df.rename({"concept_code": "code"})
    code_df.write_parquet(sample_metadata_path / "codes.parquet")

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
    main(
        parser.parse_args()
    )
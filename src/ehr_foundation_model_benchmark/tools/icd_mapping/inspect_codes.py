from pathlib import Path
import polars as pl

#input_meds = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_meds_v3_with_unit_discharge"
#codes = pl.read_parquet(Path(input_meds) / "metadata" / "codes.parquet")
#print(codes)

output_meds = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/temp_icd_conversion/"
translated_path = Path(output_meds) / "translated"

all_icd9_codes = []  # Running list to store all ICD9 codes

for parquet_file_path in translated_path.glob("*.parquet"):
    translated_data = pl.read_parquet(parquet_file_path)  # Read the current file
    icd9_codes = [code for code in translated_data['code'].to_list() if code.startswith('ICD9')]
    all_icd9_codes.extend(icd9_codes)  # Add the ICD9 codes from this file to the running list

# Print ICD9 codes that are still in the data after mapping
all_icd9_codes = list(set(all_icd9_codes))
print(all_icd9_codes)

mapping_path = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/temp_icd_conversion/metadata/vocabulary_cache/mapping"
mapping = pl.read_parquet(Path(mapping_path) / "data.parquet")

#print(mapping.filter(mapping['source_code'] == 'ICD9CM/E966.0'))
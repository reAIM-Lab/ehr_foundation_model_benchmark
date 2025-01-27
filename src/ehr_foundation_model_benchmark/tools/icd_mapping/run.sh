#!/bin/bash

# Activate Conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate MEDS

export INPUT_MEDS="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_meds_v3_with_unit_discharge"
export OUTPUT_MEDS_TEMP="/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/temp_icd_conversion/"

# Generate a sample of columbia data
PYTHONPATH=./:$PYTHONPATH python sample_columbia_meds.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP
PYTHONPATH=./:$PYTHONPATH python compute_prevalences.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP 
PYTHONPATH=./:$PYTHONPATH python map_icd.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP --source_vocabulary "ICD9" --target_vocabulary "ICD10"
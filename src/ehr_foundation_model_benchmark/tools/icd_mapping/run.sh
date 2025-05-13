#!/bin/bash

# Activate Conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate env_name

export INPUT_MEDS=""
export OUTPUT_MEDS_TEMP=""

# Generate a sample of columbia data
PYTHONPATH=./:$PYTHONPATH python sample_meds.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP
PYTHONPATH=./:$PYTHONPATH python compute_prevalences.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP 
PYTHONPATH=./:$PYTHONPATH python map_icd.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP --source_vocabulary "ICD9" --target_vocabulary "ICD10"
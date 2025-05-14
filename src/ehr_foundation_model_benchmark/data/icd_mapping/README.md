# Instructions on run ICD Mapping procedure
Converts ICD codes in input MEDs data.

## Set up the environment
```bash
conda create -n icd_mapping python=3.10
export PROJECT_ROOT=$(git rev-parse --show-toplevel)
export ICD_MAPPING_HOME="$PROJECT_ROOT/src/ehr_foundation_model_benchmark/data/icd_mapping"
```
Install the FOMO project
```bash
conda activate icd_mapping
# Install the FOMO project
pip install -e $PROJECT_ROOT
```

## Running the ICD code conversion
Set up the environment variables
```bash
export INPUT_MEDS=""
export OUTPUT_MEDS_TEMP=""
```
Running the conversion
```shell
sh $ICD_MAPPING_HOME/run.sh \
  --input_meds $INPUT_MEDS \
  --output_meds_temp $OUTPUT_MEDS_TEMP
```

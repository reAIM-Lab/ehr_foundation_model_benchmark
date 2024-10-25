export INPUT_MEDS=$1
export OUTPUT_MEDS_TEMP="$2/temp"
export OUTPUT_MEDS="$2/meds_sample"
export OUTPUT_MEDS_READER="$2/meds_sample_reader"

# Generate a sample of columbia data
PYTHONPATH=./:$PYTHONPATH python sample_columbia_meds.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP --sample_size 5000

# Combine unit with code for numeric events
pip install MEDS_transforms
export COLUMBIA_MEDS_SAMPLE=$OUTPUT_MEDS_TEMP
export COLUMBIA_MEDS_SAMPLE_UNIT_CONCATENATED=$OUTPUT_MEDS
MEDS_transform-runner "pipeline_config_fp=transform_columbia_meds_sample.yaml"
cp -r $OUTPUT_MEDS_TEMP/metadata $OUTPUT_MEDS

# Run meds_reader
pip install meds_reader
meds_reader_convert $OUTPUT_MEDS $OUTPUT_MEDS_READER
#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --input_meds INPUT_MEDS --output_meds_temp OUTPUT_MEDS_TEMP"
    echo ""
    echo "Required Arguments:"
    echo "  --input_meds PATH        Input MEDS data path"
    echo "  --output_meds_temp PATH  Temporary output MEDS data path"
    echo ""
    echo "Example:"
    echo "  $0 --input_meds /path/to/input/meds --output_meds_temp /path/to/output/temp"
    exit 1
}

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    usage
fi

# Initialize variables
INPUT_MEDS=""
OUTPUT_MEDS_TEMP=""

# Parse command line arguments
ARGS=$(getopt -o "" --long input_meds:,output_meds_temp:,help -n "$0" -- "$@")

if [ $? -ne 0 ]; then
    usage
fi

eval set -- "$ARGS"

while true; do
    case "$1" in
        --input_meds)
            INPUT_MEDS="$2"
            shift 2
            ;;
        --output_meds_temp)
            OUTPUT_MEDS_TEMP="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Internal error!"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_MEDS" ] || [ -z "$OUTPUT_MEDS_TEMP" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_MEDS_TEMP"

# Step 1: Sample MEDS data
SAMPLE_CMD="python -m ehr_foundation_model_benchmark.data.icd_mapping.sample_meds \
--input_meds \"$INPUT_MEDS\" \
--output_meds \"$OUTPUT_MEDS_TEMP\""

echo "Running MEDS data sampling:"
echo "$SAMPLE_CMD"
eval "$SAMPLE_CMD"

# Check if sampling was successful
if [ $? -ne 0 ]; then
    echo "Error: MEDS data sampling failed"
    exit 1
fi

# Step 2: Compute prevalences
PREVALENCE_CMD="python -m ehr_foundation_model_benchmark.data.icd_mapping.compute_prevalences \
--input_meds \"$INPUT_MEDS\" \
--output_meds \"$OUTPUT_MEDS_TEMP\""

echo "Running prevalence computation:"
echo "$PREVALENCE_CMD"
eval "$PREVALENCE_CMD"

# Check if prevalence computation was successful
if [ $? -ne 0 ]; then
    echo "Error: Prevalence computation failed"
    exit 1
fi

# Step 3: Map ICD
ICD_MAP_CMD="python -m ehr_foundation_model_benchmark.data.icd_mapping.map_icd \
--input_meds \"$INPUT_MEDS\" \
--output_meds \"$OUTPUT_MEDS_TEMP\" \
--source_vocabulary \"ICD9\" \
--target_vocabulary \"ICD10\""

echo "Running ICD mapping:"
echo "$ICD_MAP_CMD"
eval "$ICD_MAP_CMD"

# Check if ICD mapping was successful
if [ $? -ne 0 ]; then
    echo "Error: ICD mapping failed"
    exit 1
fi

echo "All steps completed successfully"
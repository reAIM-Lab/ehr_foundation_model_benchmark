#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --base_dir=DIR                 Base directory containing cohorts (required)"
    echo "  --output_dir=DIR               Output directory for results (required)"
    echo "  --dataset_prepared_path=PATH   Path to prepared dataset (required)"
    echo "  --model_path=PATH              Path to pre-trained model and tokenizer (required)"
    echo "  --preprocessing_workers=NUM    Number of preprocessing workers (required)"
    echo "  --batch_size=NUM               Batch size for evaluation (required)"
    echo "  --model_name=NAME              Name for the model output directory (default: cehrbert)"
    echo ""
    echo "Example:"
    echo "  $0 --base_dir=/path/to/cohorts --output_dir=/path/to/output --dataset_prepared_path=/path/to/dataset_prepared \\"
    echo "     --model_path=/path/to/model --preprocessing_workers=16 --batch_size=64 \\"
    echo "     --model_name=my_model"
    exit 1
}

# Default values
MODEL_NAME="cehrbert"

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --base_dir=*)
            BASE_DIR="${arg#*=}"
            ;;
        --output_dir=*)
            OUTPUT_DIR="${arg#*=}"
            ;;
        --dataset_prepared_path=*)
            DATASET_PREPARED_PATH="${arg#*=}"
            ;;
        --model_path=*)
            MODEL_PATH="${arg#*=}"
            ;;
        --preprocessing_workers=*)
            PREPROCESSING_WORKERS="${arg#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            ;;
        --model_name=*)
            MODEL_NAME="${arg#*=}"
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Error: Unknown option: $arg"
            usage
            ;;
    esac
done

# Check for required arguments
if [ -z "$BASE_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$DATASET_PREPARED_PATH" ] || [ -z "$MODEL_PATH" ] || [ -z "$PREPROCESSING_WORKERS" ] || [ -z "$BATCH_SIZE" ]; then
    echo "Error: Missing required arguments"
    usage
fi

# Validate arguments
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory does not exist: $BASE_DIR"
    exit 1
fi

if [ ! -d "$DATASET_PREPARED_PATH" ]; then
    echo "Error: Dataset prepared path does not exist: $DATASET_PREPARED_PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if preprocessing workers is a number
if ! [ "$PREPROCESSING_WORKERS" -eq "$PREPROCESSING_WORKERS" ] 2>/dev/null; then
    echo "Error: Preprocessing workers must be a number: $PREPROCESSING_WORKERS"
    exit 1
fi

# Check if batch size is a number
if ! [ "$BATCH_SIZE" -eq "$BATCH_SIZE" ] 2>/dev/null; then
    echo "Error: Batch size must be a number: $BATCH_SIZE"
    exit 1
fi

# Log file setup
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

# Log function
log() {
    message="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$message" | tee -a "$MAIN_LOG"
}

# Main execution
log "Starting feature extraction and model training process"
log "Configuration:"
log "  --base_dir=$BASE_DIR"
log "  --output_dir=$OUTPUT_DIR"
log "  --dataset_prepared_path=$DATASET_PREPARED_PATH"
log "  --model_path=$MODEL_PATH"
log "  --preprocessing_workers=$PREPROCESSING_WORKERS"
log "  --batch_size=$BATCH_SIZE"
log "  --model_name=$MODEL_NAME"

# Find valid cohorts and write to a temp file
TEMP_COHORT_LIST="$LOG_DIR/cohort_list_${TIMESTAMP}.txt"
> "$TEMP_COHORT_LIST" # Clear the file

# Find all valid cohorts (directories with train and test subdirectories)
for cohort_dir in "$BASE_DIR"/*; do
    if [ -d "$cohort_dir" ] && [ -d "$cohort_dir/train" ] && [ -d "$cohort_dir/test" ]; then
        cohort_name=$(basename "$cohort_dir")
        echo "$cohort_name" >> "$TEMP_COHORT_LIST"
    fi
done

# Check if any valid cohorts were found
if [ ! -s "$TEMP_COHORT_LIST" ]; then
    log "ERROR: No valid cohorts found in $BASE_DIR"
    rm -f "$TEMP_COHORT_LIST"
    exit 1
fi

# Display all cohorts that will be processed
cohort_count=$(wc -l < "$TEMP_COHORT_LIST")
log "Found $cohort_count cohorts to process:"
while read -r cohort; do
    log "- $cohort"
done < "$TEMP_COHORT_LIST"

# Process each cohort sequentially
while read -r cohort_name; do
    cohort_dir="$BASE_DIR/$cohort_name"
    output_dir="$OUTPUT_DIR/$cohort_name/$MODEL_NAME"

    log "===================================================="
    log "Processing cohort: $cohort_name"
    log "===================================================="

    cohort_log="$LOG_DIR/${cohort_name}_${TIMESTAMP}.log"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"

    # Step 1: Feature extraction
    log "Starting feature extraction for $cohort_name..."
    log "Command: python -u -m cehrbert.linear_prob.compute_cehrbert_features --data_folder $cohort_dir/train/ --test_data_folder $cohort_dir/test/ --dataset_prepared_path $DATASET_PREPARED_PATH --model_name_or_path $MODEL_PATH --tokenizer_name_or_path $MODEL_PATH --output_dir $output_dir --preprocessing_num_workers $PREPROCESSING_WORKERS --per_device_eval_batch_size $BATCH_SIZE"

    python -u -m cehrbert.linear_prob.compute_cehrbert_features \
        --data_folder "$cohort_dir/train/" \
        --test_data_folder "$cohort_dir/test/" \
        --dataset_prepared_path "$DATASET_PREPARED_PATH" \
        --model_name_or_path "$MODEL_PATH" \
        --tokenizer_name_or_path "$MODEL_PATH" \
        --output_dir "$output_dir" \
        --preprocessing_num_workers "$PREPROCESSING_WORKERS" \
        --per_device_eval_batch_size "$BATCH_SIZE" \
        --sample_packing \
        --max_tokens_per_batch 32768 \
        > "$cohort_log" 2>&1

    feature_extraction_status=$?
    if [ $feature_extraction_status -ne 0 ]; then
        log "ERROR: Feature extraction failed for $cohort_name. Check $cohort_log for details."
        continue
    fi

    # Step 2: Model training
    log "Starting model training for $cohort_name..."
    log "Command: python -u -m cehrbert.linear_prob.train_with_cehrbert_features --features_data_dir $output_dir --output_dir $output_dir"

    python -u -m cehrbert.linear_prob.train_with_cehrbert_features \
        --features_data_dir "$output_dir" \
        --output_dir "$output_dir" \
        >> "$cohort_log" 2>&1

    model_training_status=$?
    if [ $model_training_status -ne 0 ]; then
        log "ERROR: Model training failed for $cohort_name. Check $cohort_log for details."
        continue
    fi

    # Step 3: Run evaluation
    log "Running meds-evaluation for logistic regression for $cohort_name..."

    if [ -f "$output_dir/logistic/results.json" ]; then
        log "Skipping evaluation - results.json already exists"
    elif [ -d "$output_dir/logistic/test_predictions" ]; then
        meds-evaluation-cli predictions_path="$output_dir/logistic/test_predictions" \
          output_dir="$output_dir/logistic/" \
          >> "$cohort_log" 2>&1

        eval_status=$?
        if [ $eval_status -ne 0 ]; then
            log "ERROR: Running meds-evaluation failed for logistic regression for task $cohort_name"
        else
            log "Evaluation completed successfully"
        fi
    else
        log "WARNING: No test_predictions directory found for $cohort_name. Skipping evaluation."
    fi

    log "Successfully completed processing for $cohort_name"
done < "$TEMP_COHORT_LIST"

# Clean up
rm -f "$TEMP_COHORT_LIST"

log "All processing completed"
#!/bin/sh
LOGFILE="$(dirname "$0")/run_motor_output_cbs.log"
exec > >(tee -a "$LOGFILE") 2>&1

# Default values
SCRIPT_NAME=$(basename "$0")
NUM_PROC=10
TOKENS_PER_BATCH=8192
# Use empty value to indicate no observation window specified
OBSERVATION_WINDOW=""
# Default to false for linear interpolation
USE_LINEAR_INTERPOLATION=false



# Function to display help
show_help() {
    echo "Usage: $SCRIPT_NAME COHORT_BASE_DIR [OPTIONS]"
    echo
    echo "Run MOTOR generation and fine-tuning for all cohorts in the specified directory."
    echo
    echo "Arguments:"
    echo "  COHORT_BASE_DIR          Base directory containing prediction task subdirectories"
    echo
    echo "Options:"
    echo "  -h, --help               Display this help message and exit"
    echo "  --pretraining_data       Override PRETRAINING_DATA environment variable, load ontology"
    echo "  --meds_reader            Override OMOP_MEDS_READER environment variable"
    echo "  --num_proc               Number of processors to use (default: 10)"
    echo "  --tokens_per_batch       Tokens per batch (default: 231072)"
    echo "  --observation_window     Observation window in days (optional integer value)"
    echo "  --linear_interpolation   Enable linear interpolation for the model"
    echo
    echo "Environment Variables:"
    echo "  PRETRAINING_DATA         Path to pretraining data (required if not set with --pretraining_data)"
    echo "  OMOP_MEDS_READER         Path to OMOP MEDS reader (required if not set with --meds_reader)"
    echo
    echo "Example:"
    echo "  $SCRIPT_NAME /path/to/cohorts"
    echo "  $SCRIPT_NAME /path/to/cohorts --pretraining_data /path/to/pretraining --meds_reader /path/to/reader --num_proc 8"
    echo "  $SCRIPT_NAME /path/to/cohorts --observation_window 30"
    echo "  $SCRIPT_NAME /path/to/cohorts --linear_interpolation"
}

# Parse command line options
PRETRAINING_DATA_ARG=""
OMOP_MEDS_READER_ARG=""
COHORT_BASE_DIR=""
DEVICE="cuda"

while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --pretraining_data)
            PRETRAINING_DATA_ARG="$2"
            shift 2
            ;;
        --meds_reader)
            OMOP_MEDS_READER_ARG="$2"
            shift 2
            ;;
        --model_type) MODEL_TYPE="$2"; shift 2 ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --num_proc)
            NUM_PROC="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --tokens_per_batch)
            TOKENS_PER_BATCH="$2"
            shift 2
            ;;
        --observation_window)
            OBSERVATION_WINDOW="$2"
            shift 2
            ;;
        --task)
            TASK_LIST="$2"
            shift 2
            ;;
        --min_subjects_per_batch)
            MIN_SUBJECTS_PER_BATCH="$2"
            shift 2
            ;;
        --ontology_path)
            ONTOLOGY_PATH="$2"
            shift 2
            ;;
        --main_split_path)
            MAIN_SPLIT_PATH="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --loss_type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Try '$SCRIPT_NAME --help' for more information." >&2
            exit 1
            ;;
        *)
            # First non-option argument is the cohort base directory
            if [ -z "$COHORT_BASE_DIR" ]; then
                COHORT_BASE_DIR="$1"
            else
                echo "Error: Unexpected argument: $1" >&2
                echo "Try '$SCRIPT_NAME --help' for more information." >&2
                exit 1
            fi
            shift
            ;;
    esac
done

# Check if cohort base directory was provided
if [ -z "$COHORT_BASE_DIR" ]; then
    echo "Error: Missing required argument COHORT_BASE_DIR" >&2
    echo "Try '$SCRIPT_NAME --help' for more information." >&2
    exit 1
fi

# Use command line arguments if provided, otherwise use environment variables
if [ -n "$PRETRAINING_DATA_ARG" ]; then
    PRETRAINING_DATA="$PRETRAINING_DATA_ARG"
fi

if [ -n "$OMOP_MEDS_READER_ARG" ]; then
    OMOP_MEDS_READER="$OMOP_MEDS_READER_ARG"
fi

# Check if the required variables are set
if [ -z "$PRETRAINING_DATA" ] || [ -z "$OMOP_MEDS_READER" ]; then
    echo "Error: PRETRAINING_DATA or OMOP_MEDS_READER are not set." >&2
    echo "Set them as environment variables or use --pretraining_data and --meds_reader options." >&2
    echo "Try '$SCRIPT_NAME --help' for more information." >&2
    exit 1
fi

# Check if the cohort base directory exists
if [ ! -d "$COHORT_BASE_DIR" ]; then
    echo "Error: Cohort base directory does not exist: $COHORT_BASE_DIR" >&2
    exit 1
fi

echo "Using configuration:"
echo "  COHORT_BASE_DIR: $COHORT_BASE_DIR"
echo "  PRETRAINING_DATA: $PRETRAINING_DATA"
echo "  OMOP_MEDS_READER: $OMOP_MEDS_READER"
echo "  NUM_PROC: $NUM_PROC"
echo "  TOKENS_PER_BATCH: $TOKENS_PER_BATCH"
echo "  OBSERVATION_WINDOW: $([ -z "$OBSERVATION_WINDOW" ] && echo "Not specified" || echo "$OBSERVATION_WINDOW")"
echo "  USE_LINEAR_INTERPOLATION: $USE_LINEAR_INTERPOLATION"
echo

# Iterate over all task directories in the cohort folder
echo "Discovering prediction tasks..."
TASK_COUNT=0

for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    # Skip if not a directory
    if [ ! -d "$TASK_DIR" ]; then
        continue
    fi

    # Extract task name (directory name)
    TASK_NAME=$(basename "$TASK_DIR")
    TASK_COUNT=$((TASK_COUNT + 1))

    echo "[$TASK_COUNT] Found task: $TASK_NAME"
done

if [ "$TASK_COUNT" -eq 0 ]; then
    echo "No prediction tasks found in $COHORT_BASE_DIR"
    exit 0
fi

# echo "Found $TASK_COUNT prediction tasks."
# echo

# Process tasks
CURRENT=0
for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    # Skip if not a directory
    if [ ! -d "$TASK_DIR" ]; then
        continue
    fi

    # Extract task name (directory name)
    TASK_NAME=$(basename "$TASK_DIR")

    # Check if task list is specified and if current task is in the list
    if [ -n "$TASK_LIST" ]; then
        # Convert comma-separated list to array and check if task is in it
        if ! echo ",$TASK_LIST," | grep -q ",$TASK_NAME,"; then
            SKIPPED=$((SKIPPED + 1))
            echo "[$SKIPPED skipped] Skipping task: $TASK_NAME (not in specified task list)"
            continue
        fi
    fi

    CURRENT=$((CURRENT + 1))

    echo "[$CURRENT/$TASK_COUNT] Processing task: $TASK_NAME"
    echo "cohort_base_dir: $COHORT_BASE_DIR"
    echo "Task directory: $TASK_DIR"



    # Run the first command: generate MOTOR features
    echo "Running $MODEL_NAME feature generation for $TASK_NAME..."

    # Build the command with conditional observation_window parameter
    # GENERATE_CMD="python -u -m femr.omop_meds_tutorial.evaluation.generate_motor_features \
    #   --pretraining_data \"$PRETRAINING_DATA\" \
    #   --model_path \"$MODEL_PATH\" \
    #   --model_name \"$MODEL_NAME\" \
    #   --meds_reader \"$OMOP_MEDS_READER\" \
    #   --num_proc \"$NUM_PROC\" \
    #   --tokens_per_batch \"$TOKENS_PER_BATCH\" \
    #   --device \"$DEVICE\" \
    #   --min_subjects_per_batch \"$MIN_SUBJECTS_PER_BATCH\" \
    #   --cohort_dir \"$TASK_DIR\" \
    #   --ontology_path \"$ONTOLOGY_PATH\" \
    #   --output_root \"$OUTPUT_DIR\" \
    #   --task_type "binary" \
    #   --loss_type "labeled_subjects" "

    # # Add linear_interpolation parameter if specified
    # if [ "$USE_LINEAR_INTERPOLATION" = true ]; then
    #     GENERATE_CMD="$GENERATE_CMD --linear_interpolation"
    # fi

    # # Add observation_window parameter if specified
    # if [ -n "$OBSERVATION_WINDOW" ]; then
    #     GENERATE_CMD="$GENERATE_CMD --observation_window \"$OBSERVATION_WINDOW\""
    # fi

    # # Print the command
    # echo "Executing command: $GENERATE_CMD"

    # # # Execute the command
    # eval $GENERATE_CMD

    # Check if the first command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: $MODEL_NAME feature generation failed for task $TASK_NAME"
        continue
    fi

    # Run the second command: fine-tune MOTOR
    echo "Running $MODEL_NAME fine-tuning for $TASK_NAME..."

    # Build the command with conditional observation_window parameter
    FINETUNE_CMD="python -u -m femr.omop_meds_tutorial.evaluation.finetune_motor \
      --cohort_label \"$TASK_NAME\" \
      --model_name \"$MODEL_NAME\" \
      --main_split_path \"$MAIN_SPLIT_PATH\" \
      --meds_reader \"$OMOP_MEDS_READER\" \
      --model_path \"$MODEL_PATH\" \
      --output_root \"$OUTPUT_DIR\""

    # Add observation_window parameter if specified
    if [ -n "$OBSERVATION_WINDOW" ]; then
        FINETUNE_CMD="$FINETUNE_CMD --observation_window \"$OBSERVATION_WINDOW\""
    fi

    # Print the command
    echo "Executing command: $FINETUNE_CMD"

    # Execute the command
    eval $FINETUNE_CMD

    # Check if the second command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: $MODEL_NAME fine-tuning failed for task $TASK_NAME"
        continue
    fi

    # # Determine the MOTOR prediction folder path based on observation window
    if [ -n "$OBSERVATION_WINDOW" ]; then
        MOTOR_PREDICTION_FOLDER="$OUTPUT_DIR/"results"/$TASK_NAME/$MODEL_NAME_$OBSERVATION_WINDOW/test_predictions"
        MOTOR_OUTPUT_DIR="$OUTPUT_DIR/"results"/$TASK_NAME/$MODEL_NAME_$OBSERVATION_WINDOW/"
    else
        MOTOR_PREDICTION_FOLDER="$OUTPUT_DIR/"results"/$TASK_NAME/$MODEL_NAME/test_predictions"
        MOTOR_OUTPUT_DIR="$OUTPUT_DIR/"results"/$TASK_NAME/$MODEL_NAME/"
    fi

    # Build the evaluation command
    EVAL_CMD="meds-evaluation-cli predictions_path=\"$MOTOR_PREDICTION_FOLDER\" \
      output_dir=\"$MOTOR_OUTPUT_DIR\""

    # Run the third command to compute the metrics
    echo "Running meds-evaluation for $TASK_NAME..."
    echo "Executing command: $EVAL_CMD"

    # Execute the command
    eval $EVAL_CMD

    # Check if the third command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Running meds-evaluation failed for task $TASK_NAME"
    fi

    echo "Completed processing of task: $TASK_NAME"
    echo "----------------------------------------"
done

echo "All tasks processed."


# phenotype
# export CUDA_VISIBLE_DEVICES=5


# bash run_motor_cbs.sh  \
#   --pretraining_data   /user/zj2398/cache/motor_mimic_8k \
#   --meds_reader        /shared/share_mala/zj2398/mimic/meds_v0.6_reader \
#   --num_proc           100 \
#   --model_path         /user/zj2398/cache/motor_mimic_8k/output/best_100620 \
#   --model_name         motor \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path       /user/zj2398/cache/motor_mimic_8k/ontology.pkl \
#   --main_split_path     /user/zj2398/cache/motor_mimic_8k/main_split.csv \
#   --output_dir   /shared/share_mala/zj2398/mimic/phenotype_task/ \
#   /shared/share_mala/zj2398/mimic/phenotype_task/cohort/

# bash run_motor_cbs.sh  \
#   --pretraining_data   /user/zj2398/cache/motor_mimic_8k \
#   --meds_reader        /shared/share_mala/zj2398/mimic/meds_v0.6_reader \
#   --num_proc           100 \
#   --model_path         /user/zj2398/cache/motor_mimic_8k/output/best_100620 \
#   --model_name         motor \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path       /user/zj2398/cache/motor_mimic_8k/ontology.pkl \
#   --main_split_path     /user/zj2398/cache/motor_mimic_8k/main_split.csv \
#   --output_dir   /shared/share_mala/zj2398/mimic/patient_outcome_task/ \
#   /shared/share_mala/zj2398/mimic/patient_outcome_task/cohort/

# export CUDA_VISIBLE_DEVICES=3
# bash run_motor_cbs.sh \
#   --pretraining_data   /user/zj2398/cache/motor_mimic_8k \
#   --meds_reader        /shared/share_mala/zj2398/mimic/meds_v0.6_reader \
#   --num_proc           64 \
#   --model_path         /user/zj2398/cache/motor_mimic_8k/output/best_100620 \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path       /user/zj2398/cache/motor_mimic_8k/ontology.pkl \
#   --main_split_path     /user/zj2398/cache/motor_mimic_8k/main_split.csv \
#   --model_name motor \
#   --loss_type  labeled_subjects \
#   /user/zj2398/cache/mimic/mimic-3.1-meds/phenotype_task/


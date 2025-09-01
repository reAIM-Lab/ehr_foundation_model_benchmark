#!/bin/sh

# Default values
SCRIPT_NAME=$(basename "$0")
# Use empty value to indicate no observation window specified
OBSERVATION_WINDOW=""

# Function to display help
show_help() {
    echo "Usage: $SCRIPT_NAME COHORT_BASE_DIR [OPTIONS]"
    echo
    echo "Run prediction tasks for all cohorts in the specified directory."
    echo
    echo "Arguments:"
    echo "  COHORT_BASE_DIR          Base directory containing prediction task subdirectories"
    echo
    echo "Options:"
    echo "  -h, --help               Display this help message and exit"
    echo "  --pretraining_data       Override PRETRAINING_DATA environment variable"
    echo "  --meds_reader            Override OMOP_MEDS_READER environment variable"
    echo "  --observation_window     Observation window in days (optional integer value)"
    echo
    echo "Environment Variables:"
    echo "  PRETRAINING_DATA         Path to pretraining data (required if not set with --pretraining_data)"
    echo "  OMOP_MEDS_READER         Path to OMOP MEDS reader (required if not set with --meds_reader)"
    echo
    echo "Example:"
    echo "  $SCRIPT_NAME /path/to/cohorts"
    echo "  $SCRIPT_NAME /path/to/cohorts --pretraining_data /path/to/pretraining --meds_reader /path/to/reader"
    echo "  $SCRIPT_NAME /path/to/cohorts --observation_window 30"
}

# Parse command line options
PRETRAINING_DATA_ARG=""
OMOP_MEDS_READER_ARG=""
COHORT_BASE_DIR=""

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
        --observation_window)
            OBSERVATION_WINDOW="$2"
            shift 2
            ;;
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
echo "  OBSERVATION_WINDOW: $([ -z "$OBSERVATION_WINDOW" ] && echo "Not specified" || echo "$OBSERVATION_WINDOW")"
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

echo "Found $TASK_COUNT prediction tasks."
echo

# Process tasks
CURRENT=0
for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    # Skip if not a directory
    if [ ! -d "$TASK_DIR" ]; then
        continue
    fi

    # Extract task name (directory name)
    TASK_NAME=$(basename "$TASK_DIR")
    CURRENT=$((CURRENT + 1))

    echo "[$CURRENT/$TASK_COUNT] Processing task: $TASK_NAME"
    echo "Task directory: $TASK_DIR"

    # Run the first command: generate tabular features
    echo "Running tabular feature generation for $TASK_NAME..."

    # Build the command with conditional observation_window parameter
    GENERATE_CMD="python -u -m femr.omop_meds_tutorial.generate_tabular_features \
      --pretraining_data \"$PRETRAINING_DATA\" \
      --meds_reader \"$OMOP_MEDS_READER\" \
      --cohort_dir \"$TASK_DIR\""

    # Add observation_window parameter if specified
    if [ -n "$OBSERVATION_WINDOW" ]; then
        GENERATE_CMD="$GENERATE_CMD --observation_window \"$OBSERVATION_WINDOW\""
    fi

    # Print the command
    echo "Executing command: $GENERATE_CMD"

    # Execute the command
    eval $GENERATE_CMD

    # Check if the first command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Feature generation failed for task $TASK_NAME"
        continue
    fi

    # Run the second command: train baseline
    echo "Running baseline training for $TASK_NAME..."

    # Build the command with conditional observation_window parameter
    TRAIN_CMD="python -u -m femr.omop_meds_tutorial.train_baseline \
      --pretraining_data \"$PRETRAINING_DATA\" \
      --meds_reader \"$OMOP_MEDS_READER\" \
      --cohort_label \"$TASK_NAME\""

    # Add observation_window parameter if specified
    if [ -n "$OBSERVATION_WINDOW" ]; then
        TRAIN_CMD="$TRAIN_CMD --observation_window \"$OBSERVATION_WINDOW\""
    fi

    # Print the command
    echo "Executing command: $TRAIN_CMD"

    # Execute the command
    eval $TRAIN_CMD

    # Check if the second command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Baseline training failed for task $TASK_NAME"
        continue
    fi

    # Determine the output folder paths based on observation window
    if [ -n "$OBSERVATION_WINDOW" ]; then
        GBM_PREDICTION_FOLDER="$PRETRAINING_DATA/results/$TASK_NAME/baseline_$OBSERVATION_WINDOW/gbm/test_predictions"
        GBM_OUTPUT_DIR="$PRETRAINING_DATA/results/$TASK_NAME/baseline_$OBSERVATION_WINDOW/gbm/"

        LOGISTIC_PREDICTION_FOLDER="$PRETRAINING_DATA/results/$TASK_NAME/baseline_$OBSERVATION_WINDOW/logistic/test_predictions"
        LOGISTIC_OUTPUT_DIR="$PRETRAINING_DATA/results/$TASK_NAME/baseline_$OBSERVATION_WINDOW/logistic/"
    else
        GBM_PREDICTION_FOLDER="$PRETRAINING_DATA/results/$TASK_NAME/baseline/gbm/test_predictions"
        GBM_OUTPUT_DIR="$PRETRAINING_DATA/results/$TASK_NAME/baseline/gbm/"

        LOGISTIC_PREDICTION_FOLDER="$PRETRAINING_DATA/results/$TASK_NAME/baseline/logistic/test_predictions"
        LOGISTIC_OUTPUT_DIR="$PRETRAINING_DATA/results/$TASK_NAME/baseline/logistic/"
    fi

    # Run the third command to compute the metrics for GBM
    echo "Running meds-evaluation for gbm for $TASK_NAME..."

    # Build the evaluation command for GBM
    GBM_EVAL_CMD="meds-evaluation-cli predictions_path=\"$GBM_PREDICTION_FOLDER\" \
      output_dir=\"$GBM_OUTPUT_DIR\""

    # Print the command
    echo "Executing command: $GBM_EVAL_CMD"

    # Execute the command
    eval $GBM_EVAL_CMD

    # Check if the GBM evaluation command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Running meds-evaluation failed for gbm task for $TASK_NAME"
    fi

    # Run the fourth command to compute the metrics for logistic regression
    echo "Running meds-evaluation for logistic regression for $TASK_NAME..."

    # Build the evaluation command for logistic regression
    LOGISTIC_EVAL_CMD="meds-evaluation-cli predictions_path=\"$LOGISTIC_PREDICTION_FOLDER\" \
      output_dir=\"$LOGISTIC_OUTPUT_DIR\""

    # Print the command
    echo "Executing command: $LOGISTIC_EVAL_CMD"

    # Execute the command
    eval $LOGISTIC_EVAL_CMD

    # Check if the logistic regression evaluation command succeeded
    if [ $? -ne 0 ]; then
        echo "Error: Running meds-evaluation failed for logistic regression for task $TASK_NAME"
    fi

    echo "Completed processing of task: $TASK_NAME"
    echo "----------------------------------------"
done

echo "All tasks processed."
#!/bin/sh
# run_motor_regression.sh
# Clean run: invoke scripts by file path (not -m) to avoid runpy warnings.
# Writes all artifacts under the directory you launch from via --output_root.

set -e

LOGFILE="$(dirname "$0")/run_motor_output.log"
exec > >(tee -a "$LOGFILE") 2>&1

SCRIPT_NAME=$(basename "$0")
NUM_PROC=10
TOKENS_PER_BATCH=8192
OBSERVATION_WINDOW=""
USE_LINEAR_INTERPOLATION=false
REGRESSION=false
DEVICE="cuda"

show_help() {
    echo "Usage: $SCRIPT_NAME COHORT_BASE_DIR [OPTIONS]"
    echo
    echo "Options:"
    echo "  --pretraining_data       Override PRETRAINING_DATA env var (ontology, splits, etc.)"
    echo "  --meds_reader            Override OMOP_MEDS_READER env var"
    echo "  --model_path             Path to MOTOR encoder checkpoint"
    echo "  --num_proc               Number of processors (default: 10)"
    echo "  --device                 Device string (default: cuda)"
    echo "  --tokens_per_batch       Tokens per batch (default: 8192)"
    echo "  --min_subjects_per_batch Minimum subjects per batch"
    echo "  --ontology_path          Path to ontology.pkl"
    echo "  --main_split_path        Path to CSV with subject_id,split"
    echo "  --observation_window     Observation window in days"
    echo "  --linear_interpolation   Enable linear interpolation"
    echo "  --regression             Use regression evaluator (RMSE/MAE/R2)"
}

PRETRAINING_DATA_ARG=""
OMOP_MEDS_READER_ARG=""
COHORT_BASE_DIR=""
MIN_SUBJECTS_PER_BATCH="1"

while [ $# -gt 0 ]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        --pretraining_data) PRETRAINING_DATA_ARG="$2"; shift 2 ;;
        --meds_reader) OMOP_MEDS_READER_ARG="$2"; shift 2 ;;
        --model_type) MEDS_TYPE="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --num_proc) NUM_PROC="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --tokens_per_batch) TOKENS_PER_BATCH="$2"; shift 2 ;;
        --observation_window) OBSERVATION_WINDOW="$2"; shift 2 ;;
        --min_subjects_per_batch) MIN_SUBJECTS_PER_BATCH="$2"; shift 2 ;;
        --ontology_path) ONTOLOGY_PATH="$2"; shift 2 ;;
        --main_split_path) MAIN_SPLIT_PATH="$2"; shift 2 ;;
        --cohort_dir) COHORT_BASE_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --regression) REGRESSION=true; shift ;;
        -*) echo "Error: Unknown option: $1" >&2; exit 1 ;;
        *)
            # if [ -z "$COHORT_BASE_DIR" ]; then
            #     COHORT_BASE_DIR="$1"; shift
            # else
            #     echo "Error: Unexpected argument: $1" >&2; exit 1
            # fi
            # ;;
    esac
done

[ -z "$COHORT_BASE_DIR" ] && { echo "Error: Missing required argument COHORT_BASE_DIR" >&2; exit 1; }

[ -n "$PRETRAINING_DATA_ARG" ] && PRETRAINING_DATA="$PRETRAINING_DATA_ARG"
[ -n "$OMOP_MEDS_READER_ARG" ] && OMOP_MEDS_READER="$OMOP_MEDS_READER_ARG"

if [ -z "$PRETRAINING_DATA" ] || [ -z "$OMOP_MEDS_READER" ]; then
    echo "Error: PRETRAINING_DATA or OMOP_MEDS_READER are not set." >&2
    exit 1
fi

[ ! -d "$COHORT_BASE_DIR" ] && { echo "Error: Cohort base directory does not exist: $COHORT_BASE_DIR" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# repo root is three levels up from this script (…/femr/omop_meds_tutorial/evaluation/regression)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUN_ROOT="$(pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export PYTHONUNBUFFERED=1

GEN_FEATURES_PY="$REPO_ROOT/femr/omop_meds_tutorial/evaluation/generate_mtpp_features.py"
FINETUNE_PY="$REPO_ROOT/femr/omop_meds_tutorial/evaluation/regression/finetune_regression.py"

[ -f "$GEN_FEATURES_PY" ] || { echo "Missing: $GEN_FEATURES_PY"; exit 1; }
[ -f "$FINETUNE_PY" ] || { echo "Missing: $FINETUNE_PY"; exit 1; }

echo "Using configuration:"
echo "  COHORT_BASE_DIR:          $COHORT_BASE_DIR"
echo "  PRETRAINING_DATA:         $PRETRAINING_DATA"
echo "  OMOP_MEDS_READER:         $OMOP_MEDS_READER"
echo "  NUM_PROC:                 $NUM_PROC"
echo "  TOKENS_PER_BATCH:         $TOKENS_PER_BATCH"
echo "  OBSERVATION_WINDOW:       $([ -z "$OBSERVATION_WINDOW" ] && echo "Not specified" || echo "$OBSERVATION_WINDOW")"
# echo "  USE_LINEAR_INTERPOLATION: $USE_LINEAR_INTERPOLATION"
echo "  REGRESSION:               $REGRESSION"
echo "  PYTHONPATH head:          $REPO_ROOT"
echo "  RUN_ROOT (outputs):       $RUN_ROOT"
echo

echo "Discovering prediction tasks..."
TASK_COUNT=0
for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    [ -d "$TASK_DIR" ] || continue
    TASK_NAME=$(basename "$TASK_DIR")
    TASK_COUNT=$((TASK_COUNT + 1))
    echo "[$TASK_COUNT] Found task: $TASK_NAME"
done
[ "$TASK_COUNT" -eq 0 ] && { echo "No prediction tasks found in $COHORT_BASE_DIR"; exit 0; }
echo "Found $TASK_COUNT prediction tasks."
echo

CURRENT=0
for TASK_DIR in "$COHORT_BASE_DIR"*/; do
    [ -d "$TASK_DIR" ] || continue
    TASK_NAME=$(basename "$TASK_DIR")
    CURRENT=$((CURRENT + 1))

    echo "[$CURRENT/$TASK_COUNT] Processing task: $TASK_NAME"
    echo "Task directory: $TASK_DIR"

    # --- Generate features (direct file run; no -m) ---
    GENERATE_CMD="python -u \"$GEN_FEATURES_PY\" \
        --pretraining_data \"$PRETRAINING_DATA\" \
        --model_path \"$MODEL_PATH\" \
        --meds_reader \"$OMOP_MEDS_READER\" \
        --num_proc \"$NUM_PROC\" \
        --tokens_per_batch \"$TOKENS_PER_BATCH\" \
        --device \"$DEVICE\" \
        --min_subjects_per_batch \"$MIN_SUBJECTS_PER_BATCH\" \
        --cohort_dir \"$TASK_DIR\" \
        --ontology_path \"$ONTOLOGY_PATH\" \
        --output_root \"$OUTPUT_DIR\" \
        --task_type "regression" "
    # --output_root \"$RUN_ROOT\""
    # [ "$USE_LINEAR_INTERPOLATION" = true ] && GENERATE_CMD="$GENERATE_CMD --linear_interpolation"
    [ -n "$OBSERVATION_WINDOW" ] && GENERATE_CMD="$GENERATE_CMD --observation_window \"$OBSERVATION_WINDOW\""

    echo "Executing: $GENERATE_CMD"
    eval $GENERATE_CMD || { echo "Error: feature generation failed for $TASK_NAME"; echo "----------------------------------------"; continue; }

    # --- Finetune regression (direct file run; no -m) ---
    FINETUNE_CMD="python -u \"$FINETUNE_PY\" \
        --pretraining_data \"$PRETRAINING_DATA\" \
        --meds_reader \"$OMOP_MEDS_READER\" \
        --cohort_label \"$TASK_NAME\" \
        --main_split_path \"$MAIN_SPLIT_PATH\" \
        --ontology_path \"$ONTOLOGY_PATH\" \
        --model_path \"$MODEL_PATH\" \
        --num_proc \"$NUM_PROC\" \
        --tokens_per_batch \"$TOKENS_PER_BATCH\" \
        --device \"$DEVICE\" \
        --min_subjects_per_batch \"$MIN_SUBJECTS_PER_BATCH\" \
        --output_root \"$OUTPUT_DIR\""
    # [ "$USE_LINEAR_INTERPOLATION" = true ] && FINETUNE_CMD="$FINETUNE_CMD --linear_interpolation"
    [ -n "$OBSERVATION_WINDOW" ] && FINETUNE_CMD="$FINETUNE_CMD --observation_window \"$OBSERVATION_WINDOW\""

    echo "Executing: $FINETUNE_CMD"
    eval $FINETUNE_CMD || { echo "Error: finetune failed for $TASK_NAME"; echo "----------------------------------------"; continue; }

    echo "Completed processing of task: $TASK_NAME"
    echo "----------------------------------------"
done

echo "All tasks processed."






# ----------------------------------------
# Example 
# ----------------------------------------
# export CUDA_VISIBLE_DEVICES=1
# bash run_regression.sh \
#   --pretraining_data   /data/models/femr_tpp/motor_bin_8 \
#   --meds_reader        /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader/ \
#   --num_proc           8 \
#   --model_path         /data/models/femr_tpp/motor_bin_8/output/motor_8192 \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path      /data/models/femr_tpp/motor_bin_8/ontology.pkl \
#   --main_split_path    /data/models/femr_tpp/motor_bin_8/main_split.csv \
#   --cohort_dir   /shared/share_mala/zj2398/mimic/regression/cohort/regression_labels_1_month
#   --output_dir   /shared/share_mala/zj2398/mimic/regression
#   --regression \

#export CUDA_VISIBLE_DEVICES=0
# bash run_regression2.sh \
#   --pretraining_data   /user/zj2398/cache/motor_mimic_8k \
#   --meds_reader        /user/zj2398/cache/mimic/meds_v0.6_reader \
#   --num_proc           64 \
#   --model_path         /user/zj2398/cache/motor_mimic_8k/output/best_100620 \
#   --tokens_per_batch   65536 \
#   --device             cuda:0 \
#   --min_subjects_per_batch 8 \
#   --ontology_path      /user/zj2398/cache/motor_mimic_8k/ontology.pkl \
#   --main_split_path    /user/zj2398/cache/motor_mimic_8k/main_split.csv \
#   --cohort_dir   /shared/share_mala/zj2398/mimic/regression/cohort/regression_labels_1_month/ \
#   --output_dir   /shared/share_mala/zj2398/mimic/regression \
#   --tasks "bilirubin,creatinine" \
#   --regression 

#


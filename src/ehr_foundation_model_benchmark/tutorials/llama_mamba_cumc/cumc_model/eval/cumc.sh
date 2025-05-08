#!/bin/bash
#SBATCH --job-name=ehrshot-eval
#SBATCH --output=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.out
#SBATCH --error=/share/pi/nigam/mwornow/hf_ehr/slurm_logs/ehrshot-eval_%A.err
#SBATCH --time=48:00:00
#SBATCH --partition=nigam-h100,nigam-a100,gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --exclude=secure-gpu-1,secure-gpu-2,secure-gpu-3,secure-gpu-4,secure-gpu-5,secure-gpu-6,secure-gpu-7

# cd ../carina
# source base.sh
# cd -

# CLI arguments
PATH_TO_DATABASE=$1
MODEL_NAME=$2
PATH_TO_CKPT=$3
BATCH_SIZE=$4
OUTPUT_DIR=$5
# DEVICE=$4

tasks=("AMI" "Celiac" "CLL" "HTN" "Ischemic_Stroke" "MASLD" "Osteoporosis" "Pancreatic_Cancer" "SLE" "T2DM")

# 1. Generate patient representations
echo "Command run: '$0 $@'" | tee /dev/stderr
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python3 cumc_embeddings.py \
        --path_to_database $PATH_TO_DATABASE \
        --task_name $task \
        --path_to_model $PATH_TO_CKPT \
        --model_name $MODEL_NAME \
        --batch_size $BATCH_SIZE \
        --output_dir $OUTPUT_DIR \
        --embed_strat last \
        --chunk_strat last \
        # --device $DEVICE
done

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate patient representations"
    exit 1
fi

# 2. Evaluate patient representations
# CKPT=$(basename "$PATH_TO_CKPT")
# CKPT="${CKPT%.*}"
# cd /share/pi/nigam/$USER/ehrshot-benchmark/ehrshot/bash_scripts/
# bash 7_eval.sh "${MODEL_NAME}_${CKPT}_chunk:last_embed:last" --ehrshot --is_use_slurm

# For debugging
# python3 ../../eval/ehrshot.py \
#     --path_to_database /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/femr/extract \
#     --path_to_labels_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark_ehrshot \
#     --path_to_features_dir /share/pi/nigam/$USER/ehrshot-benchmark/EHRSHOT_ASSETS/features_ehrshot \
#     --path_to_model /share/pi/nigam/mwornow/hf_ehr/cache/runs/archive/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-true_val=2400000000-ckpt_val=2400000000-persist.ckpt \
#     --model_name test \
#     --batch_size 4 \
#     --embed_strat last \
#     --chunk_strat last 
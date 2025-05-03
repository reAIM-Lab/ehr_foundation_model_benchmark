#!/bin/bash

# List of tasks
tasks=("death" "long_los" "readmission")
train_sizes=(1000 10000 100000)
model_type="mamba-ehrshot"

# Loop through each task and run main.py with --task
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python main.py --task "$task" --model_type $model_type

    # for train_size in "${train_sizes[@]}"; do
    #     meds-evaluation-cli \
    #         predictions_path="/data/mchome/yk3043/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/finetune_stanford/predictions/${task}/${model_type}_${train_size}.parquet" \
    #         output_dir="outputs/${task}/${train_size}"
    # done
done
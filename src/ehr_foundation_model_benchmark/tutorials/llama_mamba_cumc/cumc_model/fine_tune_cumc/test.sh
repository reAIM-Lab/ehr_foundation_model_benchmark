#!/bin/bash

## input_meds="/data/processed_datasets/processed_datasets/ehr_foundation_data//ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped"
# List of tasks
# tasks=("death" "long_los" "readmission" "Schizophrenia")
# tasks=("Celiac" "CLL" "HTN" "Ischemic_Stroke")
tasks=("readmission")
model_type=$1
model_path=$2
input_meds=$3
device=$4

# Loop through each task and run main.py with --task
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python run.py --task "$task" --model_type $model_type --model_path $model_path --input_meds $input_meds --device $device

    # for train_size in "${train_sizes[@]}"; do
    #     meds-evaluation-cli \
    #         predictions_path="/data/mchome/yk3043/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/finetune_stanford/predictions/${task}/${model_type}_${train_size}.parquet" \
    #         output_dir="outputs/${task}/${train_size}"
    # done
done
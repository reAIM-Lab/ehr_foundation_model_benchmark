#!/bin/bash

tasks=("readmission")
model_type=$1
model_path=$2
input_meds=$3
device=$4

# Loop through each task and run main.py with --task
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python main_flop.py --task "$task" --model_type $model_type --model_path $model_path --input_meds $input_meds --device $device
done
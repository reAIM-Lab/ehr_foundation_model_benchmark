#!/bin/bash

# List of tasks
tasks=("death" "long_los" "readmission")
train_sizes=(1000 10000 100000)
model_type="mamba-ehrshot"

# Loop through each task and run main.py with --task
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python main.py --task "$task" --model_type $model_type
done
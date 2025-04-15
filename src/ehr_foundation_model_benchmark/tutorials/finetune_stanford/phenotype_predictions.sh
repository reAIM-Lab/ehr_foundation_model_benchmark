#!/bin/bash

# List of tasks
tasks=("AMI" "Celiac" "CLL" "HTN" "Ischemic_Stroke" "MASLD" "Osteoporosis" "Pancreatic_Cancer" "SLE" "T2DM")

# Loop through each task and run main.py with --task
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python main.py --task "$task"

    meds-evaluation-cli \
    predictions_path="/home/yk3043@mc.cumc.columbia.edu/context_clues/predictions/${task}.parquet" \
    output_dir="outputs/${task}"
done
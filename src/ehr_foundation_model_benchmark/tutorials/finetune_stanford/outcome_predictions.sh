#!/bin/bash

# List of tasks
tasks=("afib_ischemic_stroke_updated_meds" "discharge_home_death_meds" "hospitalization_meds" "cad_cabg_updated_meds" "hf_readmission_strict_meds" "t2dm_hf_meds")

# Loop through each task and run main.py with --task
for task in "${tasks[@]}"; do
    echo "Running task: $task"
    python main.py --task "$task" --outcome true

    meds-evaluation-cli \
    predictions_path="/home/yk3043@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/finetune_stanford/predictions/${task}.parquet" \
    output_dir="outputs/${task}"
done
#!/bin/bash
#--grid_ncpus=N
#SBATCH --job-name=create_cookbook
#SBATCH --output=logs/create_cookbook_%j.out
#SBATCH --error=logs/create_cookbook_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00         # Adjust runtime as needed

# Load any required modules (if applicable)
module load hf_ehr

# Run your command
# Please change the path to the dataset config and tokenizer config
python create_cookbook.py \
    --dataset meds_mimic4 \
    --path_to_dataset_config .../configs/data/meds_mimic4.yaml \
    --path_to_tokenizer_config .../configs/tokenizer/cookbook.yaml \
    --n_procs 256 \
    --chunk_size 10000 \
    --is_force_refresh

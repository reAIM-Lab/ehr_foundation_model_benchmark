grid_run --grid_submit=batch --grid_ncpus=32 --grid_mem=128G --grid_long \
    ./create_cookbook.py \
    --dataset meds_mimic4 \
    --path_to_dataset_config /user/zj2398/long_context_clues/hf_ehr/configs/data/meds_mimic4.yaml \
    --path_to_tokenizer_config /user/zj2398/long_context_clues/hf_ehr/configs/tokenizer/cookbook.yaml \
    --n_procs 256 \
    --chunk_size 10000 \
    --is_force_refresh

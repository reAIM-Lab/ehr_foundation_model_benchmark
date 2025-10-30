# CEHR GPT MEDS

After configuring a conda environment for CEHRGPT (see README of root folder), instead of using OMOP data, we can use the MEDS format.

## Training

Variables:
```
export CEHR_GPT_MODEL_DIR=
export CEHR_GPT_DATA_DIR=
export CEHR_GPT_PREPARED_DATA_DIR=
```

Training command:
```
python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner \
  --model_name_or_path $CEHR_GPT_MODEL_DIR \
  --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
  --output_dir $CEHR_GPT_MODEL_DIR \
  --data_folder "$CEHR_GPT_DATA_DIR" \
  --dataset_prepared_path "$CEHR_GPT_PREPARED_DATA_DIR" \
  --do_train true --seed 42 \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --hidden_size 768 --num_hidden_layers 14 --max_position_embeddings 2048 \
  --evaluation_strategy epoch --save_strategy epoch \
  --sample_packing --max_tokens_per_batch 16384 \
  --warmup_steps 500 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.0002 \
  --use_early_stopping --early_stopping_threshold 0.001 --is_data_in_meds --inpatient_att_function_type day --att_function_type day --include_inpatient_hour_token true --include_auxiliary_token true --include_demographic_prompt true --disconnect_problem_list_events true --meds_to_cehrbert_conversion_type "MedsToCehrbertOMOP" --meds_exclude_tables measurement observation device_exposure  --load_best_model_at_end true
```

## Inference

Specify the paths in `config.example.yaml` and then run `run.py`.
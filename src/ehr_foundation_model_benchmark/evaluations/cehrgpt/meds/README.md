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
  --hidden_size 768 --num_hidden_layers 14 --n_head 12 --max_position_embeddings 2048 \
  --evaluation_strategy epoch --save_strategy epoch \
  --sample_packing --max_tokens_per_batch 16384 \
  --warmup_steps 500 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.0002 \
  --min_prevalence  0.000042 \
  --bf16 true \
  --use_early_stopping --early_stopping_threshold 0.001 --is_data_in_meds --inpatient_att_function_type day --att_function_type day --include_inpatient_hour_token true --include_auxiliary_token true --include_demographic_prompt true --disconnect_problem_list_events true --meds_to_cehrbert_conversion_type "MedsToCehrbertOMOP" --meds_exclude_tables measurement observation device_exposure  --load_best_model_at_end true
```

## Finetuning

Unlike linear probing (see *Inference* below), finetuning updates the CEHR-GPT backbone jointly with a classification head, with Optuna hyperparameter tuning over the learning rate and a backbone/head learning-rate ratio.

### (a) Install

The hyperparameter-tuning / finetuning entrypoint is **not** in the upstream `cehrgpt` package. Install the pinned fork instead of `pip install cehrgpt`:

```bash
pip install "git+https://github.com/florian6973/cehrgpt.git@6a373e0"
# or editable:
# git clone https://github.com/florian6973/cehrgpt.git
# cd cehrgpt && git checkout 6a373e0 && pip install -e .
```

### (b) Finetuning command (grid search)

Shared variables:
```bash
export CEHR_GPT_MODEL_DIR=            # pretrained CEHR-GPT model dir
export CEHR_GPT_PREPARED_DATA_DIR=    # dataset_prepared cache
export CEHR_GPT_DATA_DIR=             # MEDS data folder (meds_reader)
export TOKENIZED_FULL_DATASET_PATH=   # optional: pre-tokenized dataset cache to skip re-tokenization
export CEHR_GPT_MODEL_DIR_finetuned=  # output dir for the finetuned model
export COHORT_DIR=                    # MEDS cohort folder for the task (set per task in (c))
export EXTRA_ARGS=""                  # task-specific flags (set per task in (c))
```

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m cehrgpt.runners.hf_cehrgpt_finetune_runner \
  --model_name_or_path "$CEHR_GPT_MODEL_DIR" --tokenizer_name_or_path "$CEHR_GPT_MODEL_DIR" \
  --output_dir "$CEHR_GPT_MODEL_DIR_finetuned" \
  --data_folder "$CEHR_GPT_DATA_DIR" --dataset_prepared_path "$CEHR_GPT_PREPARED_DATA_DIR" \
  --tokenized_full_dataset_path "$TOKENIZED_FULL_DATASET_PATH" \
  --cohort_folder "$COHORT_DIR" \
  --do_train --do_predict --seed 42 \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --hidden_size 768 --num_hidden_layers 14 --n_head 12 --max_position_embeddings 2048 \
  --evaluation_strategy epoch --save_strategy epoch \
  --sample_packing --max_tokens_per_batch 16384 \
  --warmup_ratio 0.1 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.0002 \
  --min_prevalence 0.000042 --bf16 true \
  --use_early_stopping --early_stopping_threshold 0.001 \
  --is_data_in_meds \
  --inpatient_att_function_type day --att_function_type day \
  --include_inpatient_hour_token true --include_auxiliary_token true \
  --include_demographic_prompt true --disconnect_problem_list_events true \
  --meds_to_cehrbert_conversion_type "MedsToCehrbertOMOP" \
  --meds_exclude_tables measurement observation device_exposure \
  --load_best_model_at_end true \
  --simple_head \
  --hyperparameter_tuning True --hyperparameter_tuning_is_grid True \
  --hyperparameter_learning_rates 0.00001 0.00004 \
  --hyperparameter_lr_ratios 1 2 5 \
  --hyperparameter_weight_decays 0.01 \
  --hyperparameter_batch_sizes 8 \
  --hyperparameter_num_train_epochs 10 \
  --n_trials 5 --hyperparameter_tuning_percentage 1.0 \
  $EXTRA_ARGS
```

Grid: learning rate `{1e-5, 4e-5}` × backbone/head LR ratio `{1, 2, 5}`, with `--hyperparameter_tuning_percentage 1.0` (tuned on the full finetuning set). Runs and metrics are logged to Weights & Biases.

`--simple_head` replaces CEHR-GPT's default finetuning head with a plain linear classification head — no hidden layer and no extra inputs — so the classifier matches the head used when finetuning the other benchmark models, keeping the comparison fair.

With `--do_predict`, the best model's test-set predictions are written directly to `$CEHR_GPT_MODEL_DIR_finetuned/test_predictions/*.parquet` (columns `subject_id`, `prediction_time`, `predicted_boolean_probability`, `boolean_value`), and the computed metrics to `$CEHR_GPT_MODEL_DIR_finetuned/test_results.json`. These predictions are used directly for evaluation — no separate feature-extraction or linear-probing step is needed (unlike *Inference* below).

### (c) Dataset- and task-specific settings

Set `COHORT_DIR` to the task's MEDS cohort folder, and set `EXTRA_ARGS` for the tasks that need it:

**CUMC** — phenotype tasks use a 2-year feature window, so **Ischemic Stroke** needs `--observation_window 730`; Schizophrenia and Long LOS take no extra flags.
```bash
export EXTRA_ARGS="--observation_window 730"   # Ischemic Stroke
export EXTRA_ARGS=""                            # Schizophrenia, Long LOS
```

**MIMIC** — **MASLD** and **Stroke** need `--preprocessing_batch_size 100` (avoids an out-of-bounds error during preprocessing); Readmission takes no extra flags.
```bash
export EXTRA_ARGS="--preprocessing_batch_size 100"   # MASLD, Stroke
export EXTRA_ARGS=""                                  # Readmission
```

## Inference

Specify the paths in `config.example.yaml` and then run `run.py`.
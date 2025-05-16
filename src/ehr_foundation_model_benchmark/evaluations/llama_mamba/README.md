
# Implementation of Llama & Mamba Model

This repo is mainly adopted from https://github.com/som-shahlab/hf_ehr/blob/main/README.md. The original paper is [**Context Clues paper**](https://arxiv.org/abs/2412.16178). 

```bash
conda create -n hf_env python=3.10
export PROJECT_ROOT=$(git rev-parse --show-toplevel)
export LLAMA_MAMBA_HOME="$PROJECT_ROOT/src/ehr_foundation_model_benchmark/evaluations/llama_mamba"
```

Install hf_env and the evaluation packages
```bash
conda create -n hf_env python=3.10 -y
conda activate hf_env
pip install flash-attn --no-build-isolation
# Install the package for the context clues models
pip install $LLAMA_MAMBA_HOME/hf_ehr-0.1.3-py3-none-any.whl
# Install meds-evaluation
pip install $LLAMA_MAMBA_HOME/meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl
# Install the FOMO project
pip install -e $PROJECT_ROOT
```

Let's set up some environment variables
```bash
export dataset_name=""
export path_to_dataset_config_file=""
export path_to_tokenizer_file=""
export path_to_cache_dir=""
export path_to_tokenizer_json_file=""
export num_process=32
export config_path=""
```

Step 1. Pretrain Llama & Mamba Model
------------------------

The pretraining consists of three parts: Dataset preparation, tokenizer creation and model training.

The customized EHR dataset should be converted to either [**MEDS data standard**](https://github.com/Medical-Event-Data-Standard/) or [**FEMR package**](https://github.com/som-shahlab/femr).

For tokenizer creation, please see https://github.com/som-shahlab/hf_ehr/blob/main/hf_ehr/tokenizers/README.md in details. An example for using the cookbook tokenizer is:


```bash

python -u -m hf_ehr.tokenizers.create_cookbook \
--dataset $dataset_name \
--path_to_dataset_config $path_to_dataset_config_file \
--path_to_tokenizer_config $path_to_tokenizer_file \
--path_to_extract $path_to_meds_reader_file \
--path_to_cache_dir $path_to_cache_dir \
--path_to_config $path_to_tokenizer_json_file \
--n_procs $num_process \
--chunk_size 10000 \
--is_force_refresh  
```

For example, it could be

```bash
python -u -m hf_ehr.tokenizers.create_cookbook \
--dataset MEDSDataset \
--path_to_dataset_config "hf_ehr/configs/data/meds_mimic4.yaml" \
--path_to_tokenizer_config "hf_ehr/configs/tokenizer/cookbook.yaml" \
--path_to_extract "hf_ehr/data/meds_reader" \
--path_to_cache_dir "hf_ehr/cache/create_cookbook" \
--path_to_config "hf_ehr/cache/tokenizers/cookbook/tokenizer_config.json" \
--n_procs 64 \
--chunk_size 10000 \
--is_force_refresh
```

You need to specify the path to preprocessed dataset, path to yaml file of tokenizer etc. You can change .yaml file to determine the path for storing tokenizer vocabulary file

Then, you can launch a Llama run on the preprocessed dataset and tokenizer (using `run.py`):
```bash
python -m hf_ehr.scripts.run \
	--config-dir=$config_path \
    +data=$dataset_config_name \
    +trainer=$trainer_config_name \
    +model=$model_config_name \
    +tokenizer=$tokenizer_config_name \
    data.tokenizer.path_to_config=$path_to_tokenizer_json_file \
    data.dataloader.mode=approx \
    data.dataloader.approx_batch_sampler.max_tokens=$maximum_tokens_per_batch \
    data.dataloader.max_length=$maximum_context_length \
    data.dataset.path_to_meds_reader_extract=$path_to_meds_reader_file \
    trainer.devices=$gpu_device \
    logging.wandb.name=$wandb_run_name \
    main.is_force_restart=True \
    main.path_to_output_dir=$output_dir
```

For example, it could be
```bash
python -m hf_ehr.scripts.run \
	--config-dir="hf_ehr/configs/" \
    +data=meds_mimic4 \
    +trainer=multi_gpu_4 \
    +model=llama-base \
    +tokenizer=cookbook \
    data.tokenizer.path_to_config="hf_ehr/cache/tokenizers/cookbook_39k/tokenizer_config.json" \
    data.dataloader.mode=approx \
    data.dataloader.approx_batch_sampler.max_tokens=16384 \
    data.dataloader.max_length=8192 \
    data.dataset.path_to_meds_reader_extract="hf_ehr/data/meds_reader" \
    trainer.devices=[3] \
    logging.wandb.name=mimic4-llama-run \
    main.is_force_restart=True \
    main.path_to_output_dir="hf_ehr/cache/runs/llamba"
```

Step 2. Model Evaluation of Llama & Mamba
------------------------
After pretraining, we can evaluate our models on downstream tasks. Here we define two types of tasks: phenotype and patient outcome prediction.
The first step is to extract patients' embeddings with our pretrained model. We provide two .sh files
If run all tasks at once, you can directly call outcome.sh or phenotype.sh. Eg:

```bash
sh $LLAMA_MAMBA_HOME/outcome.sh \
    --$model_type="llama" \
    --$model_path=$Model_checkpoint_path \
    --$input_meds=$meds_dir \
    --$device="cuda:0"
```

Otherwise you can generating embeddings for specific task

```bash
python $LLAMA_MAMBA_HOME/main.py \
    --model_type="mamba"
    --model_path=$Model_checkpoint_path \
    --input_meds=$meds_dir \
    --task="AMI" \
    --device="cuda:0" \
    --seed 123
```

The file structure of meds_dir is as follows. It includes both original meds datasets and task_labels for downstream evaluation.

```
$meds_dir
├── post_transform             # Meds dir after preprocessing    
    ├──data                    # contain medsdata
        ├── train 
        ├── held_out
        ├── test
    ├──metadata                # contain meds medatadata
├── task_labels     # Main script
    ├── patient_outcomes_sample
        ├── death
            ├── train.parquet  
            ├── tuning.parquet
            ├── held_out.parquet 
        ├── readmission
        ...
        ├── results   
    ├── phenotype_sample
        ├── AMI
        ├── Celiac
        ...
        ├── results
```
Note:  You can adapt `utils.py` if using a different model (Change global `BATCH_SIZE` and `CONTEXT_LENGTH` parameters).

Step 3. Evaluate the using context clue model features
------------------------
After generating patients' embeddings, we use the following script to train logistic regression classifiers with 5-fold cross-validation using scikit-learn. 
This includes few-shot experiments with varying training set sizes: 100, 1,000, 10,000, and the full training set, evaluated on a fixed test set:

```bash
sh $PROJECT_ROOT/src/ehr_foundation_model_benchmark/linear_prob/run_linear_prob_with_few_shots.sh \
  --base_dir $model_fearures_dir \
  --output_dir $output_dir \
  --meds_dir $meds_dir \
  --model_name mamba or llama
```


# Implementation of Llama & Mamba Model

This repo is mainly adopted from https://github.com/som-shahlab/hf_ehr/blob/main/README.md. The original paper is [**Context Clues paper**](https://arxiv.org/abs/2412.16178). 



```
Step 1. Installation:
------------------------

Direct install:
```bash
pip install hf-ehr
```

For faster Mamba runs, install:
```bash
pip install mamba-ssm causal-conv1d
```

Development install:
```bash
conda create -n hf_env python=3.10 -y
conda activate hf_env
pip install -r requirements.txt --no-cache-dir
pip install -e .



```
Step 2. Pretrain Llama & Mamba Model
------------------------

The pretraining consists of three parts: Dataset preparation, tokenizer creation and model training.

The customized EHR dataset should be converted to either [**MEDS data standard**](https://github.com/Medical-Event-Data-Standard/) or [**FEMR package**](https://github.com/som-shahlab/femr).

For tokenizer creation, please see https://github.com/som-shahlab/hf_ehr/blob/main/hf_ehr/tokenizers/README.md in details. An example for using the cookbook tokenizer is:


```bash
cd hf_ehr/scripts/
python -m -u hf_ehr.tokenizers.create_cookbook  --dataset MEDSDataset --path_to_dataset_config .../hf_ehr/configs/data/meds_mimic4.yaml --path_to_tokenizer_config .../hf_ehr/configs/tokenizer/cookbook.yaml --n_procs 64 --chunk_size 10000 --is_force_refresh  
```
You need to specify the path to preprocessed dataset, path to yaml file of tokenizer etc. You can change .yaml file to determine the path for storing tokenizer vocabulary file

Then, you can launch a Llama run on the preprocessed dataset and tokenizer (using `run.py`):
```bash
cd hf_ehr/scripts/
python3 hf_ehr.scripts.run \
    +data=meds_mimic4 \
    +trainer=multi_gpu_4 \
    +model=llama-base \
    +tokenizer=cookbook_k \
    data.dataloader.mode=approx \
    data.dataloader.approx_batch_sampler.max_tokens=16384 \
    data.dataloader.max_length=8192 \
    trainer.devices=[0,1,2,3] \
    logging.wandb.name=mimic4-llama-run \
    main.is_force_restart=True \
    main.path_to_output_dir= hf_ehr/cache/runs/llama_8k
```

```
Step 3. Extract Patient Representations using Llama & mamba
------------------------

We divide our tasks into two parts: phenotype tasks and patient outcome tasks. You can cutomized your tasks to run inside two .sh files and run them with commands:

```bash
hf_ehr.fine_tune_cumc.outcome.sh \
    --$model_type
    --$model_checkpoint_ path
    --$input_meds
    --$device

hf_ehr.fine_tune_cumc.phenotype.sh \
    --$model_type
    --$model_checkpoint_ path
    --$input_meds
    --$device
```

CEHR-BERT MIMIC-IV Tutorial
==========================

Step 1. ETLing into MEDS
------------------------

The first step of running this code requires an ETL into MEDS.

First, download the required software, meds_etl

```bash
pip install meds_etl==0.3.6[cpp]
```

Then, download MIMIC-IV following the instructions on https://physionet.org/content/mimiciv/2.2/

```bash
wget -r -N -c -np --user USERNAME --ask-password https://physionet.org/files/mimiciv/2.2/
```

Finally, run the mimic ETL on that downloaded data
```bash
meds_etl_mimic physionet.org/files/mimic-iv/ mimic-iv-meds --num_proc 16 --num_shards 16 --backend cpp
```

Step 2. Converting into meds_reader
------------------------
The FEMR library uses the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. This requires a second preprocessing step

```bash
pip install meds_reader==0.1.9
```

```bash
meds_reader_convert mimic-iv-meds mimic-iv-meds-reader --num_threads 16
```

Step 3. Pre-training CEHR-BERT
------------------------
First install [CEHR-BERT](https://github.com/cumc-dbmi/cehrbert) using pip 
```bash
pip install cehr-bert
```
Next, pre-train CEHR-BERT using the MIMIC data in the MEDS format
```bash
mkdir hf_cehrbert_mimic4;
mkdir dataset_prepared;
# Optional but a good idea to set the huggingface cache folder in a large disk 
# export HF_DATASETS_CACHE="/some_large_disk/huggingface_cache" 
python -m cehrbert.runners.hf_cehrbert_pretrain_runner hf_cehrbert_pretrain_config.yaml
```
Step 4. Fine-tune CEHR-BERT for specific tasks
First install [ACES](https://github.com/justin13601/ACES) using pip 
```bash
pip install es-aces
```
Download an example task
```bash
mkdir sample_config;
wget -O sample_config/first_24h.yaml https://raw.githubusercontent.com/mmcdermott/MEDS-DEV/main/src/MEDS_DEV/tasks/criteria/mortality/in_icu/first_24h.yaml
```
Then, generate a cohort using ACES
```bash
aces-cli cohort_name="first_24h" cohort_dir="sample_config" data.standard="meds" data.path="mimic-iv-meds/"
```

Next, finetune CEHR-BERT for the first_24h cohort (ICU mortality)
```bash
mkdir hf_cehrbert_mimic4_first_24h_finetuned;
mkdir dataset_prepared;
# Optional but a good idea to set the huggingface cache folder in a large disk 
# export HF_DATASETS_CACHE="/some_large_disk/huggingface_cache" 
python -m cehrbert.runners.hf_cehrbert_finetune_runner hf_cehrbert_finetune_config.yaml
```
# Columbia CEHR-GPT benchmark Pipeline
CEHR-GPT uses the OMOP data as the input directly.

```bash
conda create -n cehrgpt python=3.10
```
Install cehrbert_data, cehrgpt and the evaluation packages
```bash
conda activate cehrgpt
pip install cehrbert_data==0.0.9
pip install cehrgpt
pip install git+https://github.com/reAIM-Lab/ehr_foundation_model_benchmark.git@main
```

Let's set up some environment variables
```bash
export OMOP_DIR=""
export CEHR_GPT_DATA_DIR=""
export CEHR_GPT_MODEL_DIR=""
```
Create the dataset_prepared folder to cache the tokenized dataset
```shell
mkdir $CEHR_GPT_DATA_DIR/dataset_prepared
```

Step 1. Generate CEHR-GPT pre-training data using cehrbert_data
------------------------
We use spark as the data processing engine to generate the pretraining data. 
For that, we need to set up the relevant SPARK environment variables.
```bash
# the omop derived tables need to be built using pyspark
export SPARK_WORKER_INSTANCES="1"
export SPARK_WORKER_CORES="16"
export SPARK_EXECUTOR_CORES="2"
export SPARK_DRIVER_MEMORY="12g"
export SPARK_EXECUTOR_MEMORY="12g"
```
We generate the pretraining data using the following command, you should see a folder `patient_sequence` generated under `$CEHR_GPT_DATA_DIR`
```bash
sh src/ehr_foundation_model_benchmark/evaluations/cehrbert/create_cehrgpt_pretraining_data.sh \
  --input_folder $OMOP_DIR \
  --output_folde $CEHR_GPT_DATA_DIR \
  --start_date "1985-01-01"
```

Step 2. Pre-train CEHR-GPT
------------------------
Pretrain cehr-gpt using the data generated from the previous step
```bash
python -u -m cehrgpt.runners.hf_cehrgpt_pretrain_runner \
  --model_name_or_path $CEHR_GPT_MODEL_DIR \
  --tokenizer_name_or_path $CEHR_GPT_MODEL_DIR \
  --output_dir $CEHR_GPT_MODEL_DIR \
  --data_folder "$CEHR_GPT_DATA_DIR/patient_sequence/train" \
  --dataset_prepared_path "$CEHR_GPT_DATA_DIR/dataset_prepared" \
  --do_train true --seed 42 \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --hidden_size 768 --num_hidden_layers 14 --max_position_embeddings 2048 \
  --evaluation_strategy epoch --save_strategy epoch \
  --sample_packing --max_tokens_per_batch 16384 \
  --warmup_steps 500 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.002 \
  --use_early_stopping --early_stopping_threshold 0.001
```
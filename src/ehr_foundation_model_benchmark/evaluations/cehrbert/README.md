# CEHR-BERT benchmark Pipeline
CEHR-BERT uses the OMOP data as the input directly.

```bash
conda create -n cehrbert python=3.10
export PROJECT_ROOT=$(git rev-parse --show-toplevel)
export CEHRBERT_HOME="$PROJECT_ROOT/src/ehr_foundation_model_benchmark/evaluations/cehrbert"
```
Install cehrbert_data, cehrbert and the evaluation packages
```bash
conda activate cehrbert
pip install cehrbert_data==0.0.9
pip install cehrbert==1.4.3
# Install meds-evaluation
pip install $CEHRBERT_HOME/meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl
# Install the FOMO project
pip install -e $PROJECT_ROOT
```

Let's set up some environment variables
```bash
export OMOP_DIR=""
export CEHR_BERT_DATA_DIR=""
export CEHR_BERT_MODEL_DIR=""
# the folder that contains cehr-bert features and the corresponding labels
export CEHR_BERT_FEATURES_DIR = ""
# the folder that contains the evaluation metrics
export EVALUATION_DIR = ""
```
Create the dataset_prepared folder to cache the tokenized dataset
```shell
mkdir $CEHR_BERT_DATA_DIR/dataset_prepared
```

Step 1. Generate CEHR-BERT pre-training data using cehrbert_data
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
We generate the pretraining data using the following command, you should see a folder `patient_sequence` generated under `$CEHR_BERT_DATA_DIR`
```bash
sh $CEHRBERT_HOME/create_cehrbert_pretraining_data.sh \
  --input_folder $OMOP_DIR \
  --output_folde $CEHR_BERT_DATA_DIR \
  --start_date "1985-01-01"
```

Step 2. Pre-train CEHR-BERT
------------------------
Pretrain cehr-bert using the data generated from the previous step
```bash
python -u -m cehrbert.runners.hf_cehrbert_pretrain_runner \
  --model_name_or_path $CEHR_BERT_MODEL_DIR \
  --tokenizer_name_or_path $CEHR_BERT_MODEL_DIR \
  --output_dir $CEHR_BERT_MODEL_DIR \
  --data_folder "$CEHR_BERT_DATA_DIR/patient_sequence/train" \
  --dataset_prepared_path "$CEHR_BERT_DATA_DIR/dataset_prepared" \
  --do_train true --seed 42 \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --hidden_size 768 --num_hidden_layers 17 --max_position_embeddings 2048 \
  --evaluation_strategy epoch --save_strategy epoch \
  --sample_packing --max_tokens_per_batch 32768 \
  --warmup_steps 500 --weight_decay 0.01 \
  --num_train_epochs 50 --learning_rate 0.002 \
  --use_early_stopping --early_stopping_threshold 0.001
```

Step 3. CEHR-BERT model evaluation
------------------------
### Phenotype tasks
For patient phenotype tasks, we need to extract the patient sequences using a feature extraction window of 730 days (2 years) prior to the prediction time:
```bash
sh $CEHRBERT_HOME/extract_features_bert.sh \
  --cohort-folder $PHENOTYPE_COHORT_DIR \
  --input-dir $OMOP_DIR \
  --output-dir  "$CEHR_BERT_DATA_DIR/phenotype_cehrbert_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure" \
  --observation-window 730
```
We will run cehr-bert on the phenotype tasks
```shell
sh $CEHRBERT_HOME/run_cehrbert.sh \
  --base_dir="$CEHR_BERT_DATA_DIR/phenotype_cehrbert_sequences" \ 
  --dataset_prepared_path="$CEHR_BERT_DATA_DIR/dataset_prepared" \
     --model_path=$CEHR_BERT_MODEL_DIR \
     --output_dir=$CEHRBERT_FEATURES_DIR \
     --preprocessing_workers=8 \
     --model_name="cehrbert"
```
### Patient outcome tasks
For patient outcome prediction tasks, we extract representations using the entire patient history up to the prediction time:
```bash
sh $CEHRBERT_HOME/extract_features_bert.sh \
  --cohort-folder $PATIENT_OUTCOME_DIR \
  --input-dir $OMOP_DIR \
  --output-dir  "$CEHR_BERT_DATA_DIR/patient_outcome_cehrbert_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure"
```
We will run cehr-bert on the patient outcome tasks
```shell
sh $CEHRBERT_HOME/run_cehrbert.sh \
  --base_dir="$CEHR_BERT_DATA_DIR/patient_outcome_cehrbert_sequences" \ 
  --dataset_prepared_path="$CEHR_BERT_DATA_DIR/dataset_prepared" \
     --model_path=$CEHR_BERT_MODEL_DIR \
     --output_dir=$CEHR_BERT_FEATURES_DIR \
     --preprocessing_workers=8 \
     --model_name="cehrbert"
```

Step 4. Evaluate using CEHR-BERT features
------------------------
To evaluate model performance, we use the following script to train logistic regression classifiers with 5-fold cross-validation using scikit-learn. 
This includes few-shot experiments with varying training set sizes: 100, 1,000, 10,000, and the full training set, evaluated on a fixed test set: 
```bash
sh $PROJECT_ROOT/src/ehr_foundation_model_benchmark/tools/linear_prob/run_linear_prob_with_few_shots.sh \
  --base_dir $CEHR_BERT_FEATURES_DIR \
  --output_dir $EVALUATION_DIR \
  --meds_dir $OMOP_MEDS \
  --model_name cehrbert
```
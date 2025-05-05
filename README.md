# ehr_foundation_model_benchmark
Code repository for DBMI's EHR foundation model benchmarks.

## Requirements
Ensure you have the following installed:
- Python 3.10
- Conda for environment management
- Required Python packages (specified below)

## Models for evaluation
### EHR Foundation Models
- CORE-BEHRT
- MOTOR
- CEHR-GPT
- CEHR-BERT
- MAMBA (context clues)
- LLAMA (context clues)
### Baseline Models
- MEDS-TAB
- Logistic Regression
- Gradient Boosting Machines (GBM)

## Data Source
We used the OMOP Common Data Model (CDM) derived from electronic health record (EHR) data at Columbia University 
Irving Medical Center–NewYork-Presbyterian Hospital (CUIMC-NYP), encompassing approximately six million patient records 
with information on diagnoses, medications, procedures, and laboratory tests. We exported the OMOP domain and vocabulary 
tables from an on-premises SQL Server as Parquet files, harmonized units for all laboratory measurements, and 
mapped ICD-9 codes to ICD-10 using General Equivalence Mappings. After data cleaning, we converted the CUIMC-NYP OMOP data to 
the Medical Event Data Standard (MEDS) format to support models that require MEDS as input. 
Finally, the dataset was split into training and held-out sets using a 70:30 ratio, 
and the same patient split was used for both OMOP and MEDS datasets.
[Add more description and statistics to this section].


### Unit Harmonization
```shell

```
### ICD9 to ICD10 Mapping
```shell

```
###  Convert OMOP to MEDS
```shell
export OMOP_DIR=""
export OMOP_MEDS=""
export EVALUATION_DIR=""

conda create -n meds_etl python=3.10
conda activate meds_etl
pip install meds_etl==0.3.9
meds_etl_omop $OMOP_DIR $OMOP_MEDS --num_proc 16
```
 ### Model Summary and Input Data Formats

| Model               | Required Data Format | Description |
|---------------------|----------------------|-------------|
| CORE-BEHRT          | OMOP                 | A transformer-based model adapted from BEHRT for longitudinal patient representation using OMOP CDM structure. |
| MOTOR               | MEDS                 | A Transformer model pretrained using a piecewise exponential time-to-event objective across structured EHR sequences in MEDS format. |
| CEHR-GPT            | OMOP                 | A generative model trained on OMOP-formatted sequences to synthesize realistic patient trajectories and perform downstream predictions. |
| CEHR-BERT           | OMOP                 | A masked language model tailored for OMOP-based structured EHR data, used for patient representation and embedding learning. |
| MAMBA               | MEDS                 | A long-context architecture for EHR modeling based on state-space models, optimized for sequential MEDS-formatted data. |
| LLAMA               | MEDS                 | A large language model adapted to structured EHR inputs via MEDS, used for zero-shot and generative clinical reasoning. |
| MEDS-TAB            | MEDS                 | |
| Logistic Regression | MEDS ||
| GBM                 | MEDS ||

## Evaluation Tasks
### Phenotypes
### Patient Outcomes

## Model Evaluation
The EHR foundation models are pre-trained prior to evaluation, while the baseline models are evaluated directly without pretraining. 
Following the evaluation protocols established in MOTOR [citation] and Contexts Clues [citation], we limit our assessments to linear probing, 
as further fine-tuning of foundation models can be computationally expensive and time-consuming. For each task, 
we extract patient representations at the prediction time, train a logistic regression model using 5-fold cross-validation 
on the extracted features and corresponding labels, and report the AUROC on the held-out test. To ensure consistency, 
we use a fixed random seed for shuffling samples and fitting the logistic regression model.

### MOTOR
MOTOR is implemented in the FEMR library, which the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. 
Set up the environment
```bash
conda create -n femr python=3.10
```
Install MEDS_READER, FEMR and evaluation packages
```bash
conda activate femr
pip install meds_reader==0.0.6
pip install git+https://github.com/ChaoPang/femr.git@omop_meds_v3_tutorial
pip install git+https://github.com/reAIM-Lab/ehr_foundation_model_benchmark.git@main
```
Step 1. Pretrain MOTOR
------------------------
Follow the [Pretrain MOTOR instructions](src/ehr_foundation_model_benchmark/evaluations/motor/README.md)

Step 2. Extract patient representations using MOTOR
------------------------
Set the environment variables
```bash
# CUIMC MEDS READER folder
export OMOP_MEDS_READER = ""
# this should point to where the MOTOR data and model artifacts will be generated
export MOTOR_DIR = ""
# the folder that contains all the phenotype labels
export PHENOTYPE_COHORT_DIR = ""
# the folder that contains all the patient outcome labels
export PATIENT_OUTCOME_DIR = ""
```
For patient phenotype tasks, we extract patient representations a feature extraction window of 730 days (2 years) prior to the prediction time:
```bash
sh src/ehr_foundation_model_benchmark/evaluations/motor/run_motor.sh $PHENOTYPE_COHORT_DIR \
  --observation_window 730
```
For patient outcome prediction tasks, we extract representations using the entire patient history up to the prediction time:
```bash
sh src/ehr_foundation_model_benchmark/evaluations/motor/run_motor.sh $PATIENT_OUTCOME_DIR
```
Step 3. Evaluate using MOTOR features
------------------------
To evaluate model performance, we use the following script to train logistic regression classifiers with 5-fold cross-validation using scikit-learn. 
This includes few-shot experiments with varying training set sizes: 100, 1,000, 10,000, and the full training set, evaluated on a fixed test set: 
```bash
sh src/ehr_foundation_model_benchmark/tools/linear_prob/run_linear_prob_with_few_shots.sh \
  --base_dir $MOTOR_DIR/results/ \
  --output_dir $EVALUATION_DIR \
  --meds_dir $OMOP_MEDS \
  --model_name motor
```

### CEHR-BERT 
Set up the environment
```bash
conda create -n cehrbert python=3.10
```
Install cehrbert_data, cehrbert and the evaluation packages
```bash
conda activate cehrbert
pip install cehrbert_data==0.0.9
pip install cehrbert==1.4.3
pip install git+https://github.com/reAIM-Lab/ehr_foundation_model_benchmark.git@main
```
Let's set up some environment variables for CEHR-BERT
```bash
export OMOP_DIR=""
export CEHR_BERT_DATA_DIR=""
export CEHR_BERT_MODEL_DIR=""
```

Step 1. Pre-train CEHR-BERT
------------------------
Follow the [Pretrain CEHR-BERT instructions](src/ehr_foundation_model_benchmark/evaluations/cehrbert/README.md)

Step 2. Extract patient representations using CEHR-BERT
------------------------
For CEHR-BERT, we need to construct the patient sequences from the OMOP dataset given the task labels and prediction times,
then we use the pre-trained cehr-bert to extract the patient representation at the prediction time. 

Set the environment variables
```bash
# the folder that will store the cehr-bert sequences for patient outcome and phenotype tasks
export CEHR_BERT_DATA_DIR=""
# this should point to where the cehr-bert model artifacts will be generated
export CEHR_BERT_MODEL_DIR=""
# the folder that contains all the phenotype labels
export PHENOTYPE_COHORT_DIR = ""
# the folder that contains all the patient outcome labels
export PATIENT_OUTCOME_DIR = ""
```

### Phenotype tasks
For patient phenotype tasks, we need to extract the patient sequences using a feature extraction window of 730 days (2 years) prior to the prediction time:
```bash
sh src/ehr_foundation_model_benchmark/evaluations/cehrbert/extract_features_bert.sh \
  --cohort-folder $PHENOTYPE_COHORT_DIR \
  --input-dir $OMOP_DIR \
  --output-dir  "$CEHR_BERT_DATA_DIR/phenotype_cehrbert_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure" \
  --observation-window 730
```
We will run cehr-bert on the phenotype tasks
```shell
sh src/ehr_foundation_model_benchmark/evaluations/cehrbert/run_cehrbert.sh \
  --base_dir="$CEHR_BERT_DATA_DIR/phenotype_cehrbert_sequences" \ 
  --dataset_prepared_path="$CEHR_BERT_DATA_DIR/dataset_prepared" \
     --model_path=$CEHR_BERT_MODEL_DIR \
     --output_dir=$EVALUATION_DIR \
     --preprocessing_workers=8 \
     --model_name="cehrbert"
```
### Patient outcome tasks
For patient outcome prediction tasks, we extract representations using the entire patient history up to the prediction time:
```bash
sh src/ehr_foundation_model_benchmark/evaluations/cehrbert/extract_features_bert.sh \
  --cohort-folder $PATIENT_OUTCOME_DIR \
  --input-dir $OMOP_DIR \
  --output-dir  "$CEHR_BERT_DATA_DIR/patient_outcome_cehrbert_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure"
```
We will run cehr-bert on the patient outcome tasks
```shell
sh src/ehr_foundation_model_benchmark/evaluations/cehrbert/run_cehrbert.sh \
  --base_dir="$CEHR_BERT_DATA_DIR/patient_outcome_cehrbert_sequences" \ 
  --dataset_prepared_path="$CEHR_BERT_DATA_DIR/dataset_prepared" \
     --model_path=$CEHR_BERT_MODEL_DIR \
     --output_dir=$EVALUATION_DIR \
     --preprocessing_workers=8 \
     --model_name="cehrbert"
```

### CEHR-GPT
Set up the environment
```bash
conda create -n cehrgpt python=3.10
```
Install cehrbert_data, cehrgpt and the evaluation packages
```bash
conda activate cehrbert
pip install cehrbert_data==0.0.9
pip install cehrgpt
pip install git+https://github.com/reAIM-Lab/ehr_foundation_model_benchmark.git@main
```
Let's set up some environment variables for CEHR-GPT
```bash
export OMOP_DIR=""
export CEHR_GPT_DATA_DIR=""
export CEHR_GPT_MODEL_DIR=""
```

Step 1. Pre-train CEHR-GPT
------------------------
Follow the [Pretrain CEHR-GPT instructions](src/ehr_foundation_model_benchmark/evaluations/cehrgpt/README.md)

Step 2. Extract patient representations using CEHR-GPT
------------------------
For CEHR-GPT, we need to construct the patient sequences from the OMOP dataset given the task labels and prediction times,
then we use the pre-trained cehr-bert to extract the patient representation at the prediction time. 

Set the environment variables
```bash
# the folder that will store the cehr-bert sequences for patient outcome and phenotype tasks
export CEHR_GPT_DATA_DIR=""
# this should point to where the cehr-bert model artifacts will be generated
export CEHR_GPT_MODEL_DIR=""
# the folder that contains all the phenotype labels
export PHENOTYPE_COHORT_DIR = ""
# the folder that contains all the patient outcome labels
export PATIENT_OUTCOME_DIR = ""
```

### Phenotype tasks
For patient phenotype tasks, we need to extract the patient sequences using a feature extraction window of 730 days (2 years) prior to the prediction time:
```bash
sh src/ehr_foundation_model_benchmark/evaluations/cehrgpt/extract_features_gpt.sh \
  --cohort-folder $PHENOTYPE_COHORT_DIR \
  --input-dir $OMOP_DIR \
  --output-dir  "$CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure" \
  --observation-window 730
```
We will run cehr-gpt on the phenotype tasks
```shell
sh src/ehr_foundation_model_benchmark/evaluations/cehrgpt/run_cehrgpt.sh \
  --base_dir="$CEHR_GPT_DATA_DIR/phenotype_cehrgpt_sequences" \ 
  --dataset_prepared_path="$CEHR_GPT_DATA_DIR/dataset_prepared" \
     --model_path=$CEHR_GPT_MODEL_DIR \
     --output_dir=$EVALUATION_DIR \
     --preprocessing_workers=8 \
     --model_name="cehrgpt"
```
### Patient outcome tasks
For patient outcome prediction tasks, we extract representations using the entire patient history up to the prediction time:
```bash
sh src/ehr_foundation_model_benchmark/evaluations/cehrgpt/extract_features_gpt.sh \
  --cohort-folder $PATIENT_OUTCOME_DIR \
  --input-dir $OMOP_DIR \
  --output-dir  "$CEHR_GPT_DATA_DIR/patient_outcome_cehrgpt_sequences" \
  --patient-splits-folder "$OMOP_DIR/patient_splits" \
  --ehr-tables "condition_occurrence procedure_occurrence drug_exposure"
```
We will run cehr-gpt on the patient outcome tasks
```bash
sh src/ehr_foundation_model_benchmark/evaluations/cehrbert/run_cehrgpt.sh \
  --base_dir="$CEHR_GPT_DATA_DIR/patient_outcome_cehrgpt_sequences" \ 
  --dataset_prepared_path="$CEHR_GPT_DATA_DIR/dataset_prepared" \
  --model_path=$CEHR_GPT_MODEL_DIR \
  --output_dir=$EVALUATION_DIR \
  --preprocessing_workers=8 \
  --model_name="cehrgpt"
```
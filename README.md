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
export OMOP_FOLDER=""
export MEDS_MEDS=""
conda create -n meds_etl python=3.10
conda activate meds_etl
pip install meds_etl==0.3.9
meds_etl_omop $OMOP_FOLDER $MEDS_MEDS --num_proc 16
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
Install MEDS_READER and FEMR
```bash
conda activate femr
pip install meds_reader==0.0.6
pip install git+https://github.com/ChaoPang/femr.git@omop_meds_v3_tutorial
```
#### Step 1. Pretrain MOTOR
Follow the [Pretrain MOTOR instructions](src/ehr_foundation_model_benchmark/evaluations/motor/README.md)

#### Step 2. Extract patient representations using MOTOR
Set the environment variables
```bash
# CUIMC MEDS READER folder
export OMOP_MEDS_READER = ""
# this should point to where the MOTOR data and model artifacts will be generated
export PRETRAINING_DATA = ""
# the folder that contains all the phenotype labels
export PHENOTYPE_COHORT_DIR = ""
# the folder that contains all the patient outcome labels
export PATIENT_OUTCOME_DIR = ""
```
We extract patient representations for all the phenotypes using a feature extraction window of 2 years prior to the prediction time
```bash
sh src/femr/omop_meds_tutorial/run_motor.sh $PHENOTYPE_COHORT_DIR --observation_window 730
```
We extract patient representations for all the patient outcomes using the entire patient history prior to the prediction time
```bash
sh src/femr/omop_meds_tutorial/run_motor.sh PATIENT_OUTCOME_DIR
```

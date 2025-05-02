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

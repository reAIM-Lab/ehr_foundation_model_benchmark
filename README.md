# FoMo: A Framework for EHR Foundation Model Evaluation
Code repository for FOMO EHR foundation model benchmarks.

## Table of Contents
(Table of contents links may not due to anonymization)
- [Requirements](#requirements)
- [Models for Evaluation](#models-for-evaluation)
  - [EHR Foundation Models](#ehr-foundation-models)
  - [Baseline Models](#baseline-models)
- [Data Source](#data-source)
  - [Unit Harmonization](#unit-harmonization)
  - [ICD9 to ICD10 Mapping](#icd9-to-icd10-mapping)
  - [Convert OMOP to MEDS](#convert-omop-to-meds)
  - [Model Summary and Input Data Formats](#model-summary-and-input-data-formats)
- [Evaluation Tasks](#evaluation-tasks)
  - [Phenotypes](#phenotypes)
  - [Patient Outcomes](#patient-outcomes)
- [Model Evaluation](#model-evaluation)

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
- FEMR Baselines (Logistic Regression and LightGBM)
- MEDS-TAB (Logistic Regression and XGBoost)

## Data Source
We used the OMOP Common Data Model (CDM) derived from electronic health record (EHR) data at a large urban academic medical center, 
encompassing approximately six million patient records  with information on diagnoses, medications, procedures, and laboratory tests. 
We exported the OMOP domain and vocabulary  tables from an on-premises SQL Server as Parquet files, harmonized units for all laboratory measurements, and 
mapped ICD-9 codes to ICD-10 using General Equivalence Mappings. After data cleaning, we converted the OMOP data to 
the Medical Event Data Standard (MEDS) format to support models that require MEDS as input. 
Finally, the dataset was split into training, tuning and held-out sets using the 60:10:30 ratio, 
and the same patient split was used for both OMOP and MEDS datasets.
[Add more description and statistics to this section].


### Unit Harmonization
Detailed step-by-step instructions are available in the [Unit Harmonization Guide](src/ehr_foundation_model_benchmark/data/unit_harmonization/README.md). 
The hyperlink may not due to anonymization, please refresh the page or go to the corresponding README manually

### ICD9 to ICD10 Mapping
Follow the ICD-9 to ICD-10 mapping instructions in the [ICD Mapping Guide](src/ehr_foundation_model_benchmark/data/icd_mapping/README.md). 
The hyperlink may not due to anonymization, please refresh the page or go to the corresponding README manually

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

| Model          | Required Data Format | Description                                                                                                                                                                   |
|----------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CORE-BEHRT     | OMOP                 | A transformer-based model adapted from BEHRT for longitudinal patient representation using OMOP CDM structure.                                                                |
| MOTOR          | MEDS                 | A Transformer model pretrained using a piecewise exponential time-to-event objective across structured EHR sequences in MEDS format.                                          |
| CEHR-GPT       | OMOP                 | A generative model trained on OMOP-formatted sequences to synthesize realistic patient trajectories and perform downstream predictions.                                       |
| CEHR-BERT      | OMOP                 | A masked language model tailored for OMOP-based structured EHR data, used for patient representation and embedding learning.                                                  |
| MAMBA          | MEDS                 | A long-context architecture for EHR modeling based on state-space models, optimized for sequential MEDS-formatted data.                                                       |
| LLAMA          | MEDS                 | A large language model adapted to structured EHR inputs via MEDS, used for zero-shot and generative clinical reasoning.                                                       |
| FEMR Baselines | MEDS | A baseline model uses normalized age and event counts from a patient’s history up to prediction time, trained with Logistic Regression and LightGBM.                          |
| MEDS-TAB       | MEDS                 | MEDS-TAB extends baseline features using flexible time windows and aggregation functions (e.g., count, value, min, max), then trains Logistic Regression and XGBoost models.  |

## Evaluation Tasks
### Phenotypes
The 11 phenotypes include a diverse set of both **acute** (e.g., myocardial infarction, ischemic stroke) and **chronic** conditions (e.g., type 2 diabetes, hypertension, schizophrenia), covering a broad range of clinical domains such as cardiovascular, autoimmune, metabolic, oncologic, and psychiatric diseases. This diversity enables comprehensive evaluation of a model's ability to capture different temporal patterns, 
disease progression, and diagnostic complexity across varying prediction scenarios.

| Phenotype                          | Description |
|-----------------------------------|-------------|
| **Celiac Disease**                | A chronic autoimmune disorder triggered by gluten. Defined by diagnosis codes and risk factors like family history and digestive disorders. |
| **Acute Myocardial Infarction (AMI)** | An acute heart condition. Identified via inpatient or ER diagnoses of myocardial infarction and related ischemic heart disease indicators. |
| **Systemic Lupus Erythematosus (SLE)** | A chronic autoimmune disease. Requires a combination of symptoms, treatment history, and confirmed SLE diagnosis. |
| **Pancreatic Cancer**             | A chronic malignancy of the pancreas. Identified using pancreatic neoplasm codes, excluding benign tumors. |
| **Hypertension (HTN)**            | A chronic condition involving elevated blood pressure. Diagnosed via hypertensive disorder codes and related cardiovascular risk factors. |
| **Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD)** | A chronic liver condition formerly known as NAFLD. Defined by liver disease codes and metabolic risk factors such as obesity and diabetes. |
| **Ischemic Stroke**               | An acute neurological event due to arterial blockage. Requires I63 diagnosis codes during inpatient or ER visits. |
| **Osteoporosis**                  | A chronic bone disease leading to fragility. Case cohort based on diagnosis codes; risk factors include age, gender, body weight, and medications. |
| **Chronic Lymphoid Leukemia (CLL)** | A chronic blood cancer. Identified using CLL concept sets, with risk factors including age, symptoms, and family history. |
| **Type 2 Diabetes Mellitus (T2DM)** | A chronic metabolic disease. Requires diagnosis plus antidiabetic drug use or elevated HbA1c; risk factors include age, obesity, and prediabetes. |
| **Schizophrenia**                 | A chronic psychiatric disorder. Defined by diagnostic transition from psychosis to schizophrenia in patients aged 10–35 with adequate history. |

### Patient Outcomes
These three patient outcome tasks—**in-hospital mortality**, **30-day readmission**, and **prolonged length-of-stay**—are closely tied to hospital operations. They reflect critical quality metrics that impact resource allocation, patient safety, and institutional performance, 
making them essential for both clinical decision support and healthcare management.

| Outcome Cohort                | Description |
|-------------------------------|-------------|
| **In-hospital mortality** | Predicts whether a patient will die during a current hospitalization. Includes admissions lasting >48 hours. Prediction is made 48 hours after admission. Patients must have ≥2 years of prior observation. |
| **30-day readmission** | Predicts all-cause readmission within 30 days of discharge. Prediction time is at discharge. Patients must have ≥2 years of prior history and not be censored within 30 days post-discharge. Same-day readmissions are excluded. |
| **Prolonged length-of-stay** | Predicts whether a hospitalization lasts more than 7 days. Prediction is made 48 hours after admission. Patients must have ≥2 years of prior observation. |


## Model Evaluation
The EHR foundation models are pre-trained prior to evaluation, while the baseline models are evaluated directly without pretraining. 
Following the evaluation protocols established in [MOTOR](https://arxiv.org/abs/2301.03150) and [Contexts Clues](https://arxiv.org/abs/2412.16178), we limit our assessments to linear probing, 
as further fine-tuning of foundation models can be computationally expensive and time-consuming. For each task, 
we extract patient representations at the prediction time, train a logistic regression model using 5-fold cross-validation 
on the extracted features and corresponding labels, and report the AUROC on the held-out test. To ensure consistency, 
we use a fixed random seed for shuffling samples and fitting the logistic regression model.

The hyperlink may not due to anonymization, please refresh the page after clicking the link or go to the corresponding README manually

- MOTOR: [MOTOR Pipeline instructions](src/ehr_foundation_model_benchmark/evaluations/motor/README.md)
- CEHR-BERT: [CEHR-BERT Pipeline instructions](src/ehr_foundation_model_benchmark/evaluations/cehrbert/README.md)
- CEHR-GPT: [CEHR-GPT Pipeline instructions](src/ehr_foundation_model_benchmark/evaluations/cehrgpt/README.md)
- CORE-BEHRT: [CORE-BEHRT Pipeline instructions](src/ehr_foundation_model_benchmark/evaluations/corebehrt/README.md)
- FEMR Baselines: [FEMR Baseline Pipeline instructions](src/ehr_foundation_model_benchmark/evaluations/femr_baseline/README.md)
- MEDS-TAB Baselines: [MEDS-TAB Baseline Pipeline instructions](src/ehr_foundation_model_benchmark/evaluations/medstab/README.md)
- MAMBA-EHRSHOT Baselines: [MAMBA-EHRSHOT Pipeline instructions](src/ehr_foundation_model_benchmark/evaluations/mamba_ehrshot/README.md)
- MAMBA
- Llama

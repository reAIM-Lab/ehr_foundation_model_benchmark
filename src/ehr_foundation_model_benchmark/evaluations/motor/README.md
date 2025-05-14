# MOTOR benchmark Pipeline
MOTOR is implemented in the FEMR library, which the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. 

## Set up the environment
```bash
conda create -n femr python=3.10
```
Install MEDS_READER, FEMR and evaluation packages if you haven't done so
```bash
conda activate femr
pip install meds_reader==0.1.13
pip install femr-0.2.0-py3-none-any.whl
pip install meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl
# Install the FOMO project
pip install -e $(git rev-parse --show-toplevel)
```
Set the environment variables
```bash
# MEDS data folder
export OMOP_MEDS = ""
# MEDS READER folder
export OMOP_MEDS_READER = ""
# this should point to where the MOTOR data and model artifacts will be generated
export PRETRAINING_DATA = ""
# this needs to point to the OMOP instance, where concept.CSV, concept_ancestor.CSV, and concept_relationship.CSV are located.
# FEMR requires the CSV files rather than the parquet files
export ATHENA_DATA = ""
```

Step 1. Pretrain MOTOR
------------------------

### Converting into meds_reader
```bash
meds_reader_convert $OMOP_MEDS $OMOP_MEDS_READER --num_threads 16
```

### Downloading Athena (Optional)
FEMR uses OHDSI's Athena tool for ontology processing. Go to https://athena.ohdsi.org/ and download the folder.
You can create an account for free.
Note: Make sure to run the CPT4 fixer script in the Athena download before continuing!

### Preparing For Pretraining
We have a single script, prepare_motor that generates these splits and then training things like tokenizers to prepare for pretraining

```bash
python -u -m femr.omop_meds_tutorial.prepare_motor \
  --pretraining_data $PRETRAINING_DATA \
  --athena_path $ATHENA_DATA \
  --meds_reader $OMOP_MEDS_READER
```

### Pretrain MOTOR
You could probably also train on smaller GPUs, even 16GB but that might require some hyperparameter tweaks.

```bash
python -u -m femr.omop_meds_tutorial.pretrain_motor \
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader $OMOP_MEDS_READER
```

Step 2. Extract patient representations using MOTOR
------------------------
Set the environment variables
```bash
# MEDS READER folder
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
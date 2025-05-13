# Columbia MOTOR benchmark Pipeline
MOTOR is implemented in the FEMR library, which the [meds_reader](https://github.com/EthanSteinberg/meds_reader) utility for processing MEDS data. 

## Set up the environment
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
Set the environment variables
```bash
# CUIMC MEDS data folder
export OMOP_MEDS = ""
# CUIMC MEDS READER folder
export OMOP_MEDS_READER = ""
# this should point to where the MOTOR data and model artifacts will be generated
export PRETRAINING_DATA = ""
# this needs to point to the OMOP instance, where concept.CSV, concept_ancestor.CSV, and concept_relationship.CSV are located.
# FEMR requires the CSV files rather than the parquet files
export ATHENA_DATA = ""
```

Step 1. Converting into meds_reader
------------------------
```bash
meds_reader_convert $OMOP_MEDS $OMOP_MEDS_READER --num_threads 16
```

Step 2. Downloading Athena (Optional)
-------------------------
FEMR uses OHDSI's Athena tool for ontology processing. Go to https://athena.ohdsi.org/ and download the folder.
You can create an account for free.
Note: Make sure to run the CPT4 fixer script in the Athena download before continuing!

Step 3. Preparing For Pretraining
------------------------
We have a single script, prepare_motor that generates these splits and then training things like tokenizers to prepare for pretraining

```bash
python -u -m femr.omop_meds_tutorial.prepare_motor \
  --pretraining_data $PRETRAINING_DATA \
  --athena_path $ATHENA_DATA \
  --meds_reader $OMOP_MEDS_READER
```

Step 4. Pretrain MOTOR
------------------------
You could probably also train on smaller GPUs, even 16GB but that might require some hyperparameter tweaks.

```bash
python -u -m femr.omop_meds_tutorial.pretrain_motor \
  --pretraining_data $PRETRAINING_DATA \
  --meds_reader $OMOP_MEDS_READER
```
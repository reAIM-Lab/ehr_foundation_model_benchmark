# How to prepare units for EHR Foundation Models
### Set up the environment
```bash
conda create -n unit_harmonization python=3.10
export PROJECT_ROOT=$(git rev-parse --show-toplevel)
export UNIT_HARMONIZATION_HOME="$PROJECT_ROOT/src/ehr_foundation_model_benchmark/data/unit_harmonization"
```
Install the FOMO project
```bash
conda activate unit_harmonization
# Install the FOMO project
pip install -e $PROJECT_ROOT
pip install pymssql
```

Set up the environment variables
```bash
export SOURCE_OMOP_FOLDER=""
export HARMONIZED_OMOP_FOLDER=""
export CONCEPT_PATH=$SOURCE_OMOP_FOLDER/concept
```

Step. 1 Run `analyze_labs.py` to create the summary lab csv table
--------------------------------
We use SQL server to pull data from the OMOP measurement table, you must provide username, password, DB server and database name, 
only SQL Server Login is currently supported. This script generates the statistics associated with the measurement unit pairs in `measurement_unit_counts.csv`.  
```bash
python $UNIT_HARMONIZATION_HOME/analyze_labs.py
```

Step. 2 Run `convert_units.py` to run the conversion
--------------------------------
This step converts the measurement units using the statistics generated from `measurement_unit_counts.csv`. 
```bash
python $UNIT_HARMONIZATION_HOME/convert_units.py \
  --source_measurement_dir $SOURCE_OMOP_FOLDER/measurement
```

Step.3 Run `stats.py` to compute the conversion summary.
--------------------------------
This script processes harmonized measurement files to 
- compute majority units for each measurement concept
- classify unit and value matching types 
- Generate comprehensive statistics on unit and value harmonization
```bash
python $UNIT_HARMONIZATION_HOME/convert_units.py \
  --measurement_parquet_folder $SOURCE_OMOP_FOLDER/measurement
```

Step.4 Create a new OMOP instance with the harmonized measurement
--------------------------------
```bash
python $UNIT_HARMONIZATION_HOME/merge_harmonized_labs_with_omop.py \
  --omop_folder $SOURCE_OMOP_FOLDER \
  --harmonized_labs_folder $SOURCE_OMOP_FOLDER/measurement \
  --output_folder $HARMONIZED_OMOP_FOLDER
```
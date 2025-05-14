# CORE-BEHRT benchmark Pipeline
The CORE-BEHRT model and repo were first developed by: Mikkel Odgaard, Kiril Vadimovic Klein, Sanne Møller Thysen, Espen Jimenez-Solem, Martin Sillesen, and Mads Nielsen. Core-behrt: A carefully optimized and rigorously evaluated behrt. arXiv preprint arXiv:2404.15201, 2024

CORE-BEHRT uses the OMOP data as the input directly. The file containing patient information should be called `patients_info.csv` and contain columns for `PID`, `DATE_OF_BIRTH` and other relevant background features (such as `RACE` or `GENDER`). Event data should be stored in parquet files within the folders "condition_occurrence" and "drug_exposure". Each parquet file should have the following data fields: `TIMESTAMP`, `PID`, `ADMISSION_ID`, and `CONCEPT`. These flattened files are directly taken from OMOP data.

```bash
conda create -n corebehrt python=3.10
export PROJECT_ROOT=$(git rev-parse --show-toplevel)
export COREBEHRT_HOME="$PROJECT_ROOT/src/ehr_foundation_model_benchmark/evaluations/corebehrt"
```
Install corebehrt and the evaluation packages
```bash
conda activate corebehrt
pip install corebehrt-0.1.0-py3-none-any.whl
pip install meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl
# Install the FOMO project
pip install -e $PROJECT_ROOT
```

Let's set up some environment variables
```bash
export OMOP_DIR=""
export CORE_BEHRT_DATA_DIR=""
export CORE_BEHRT_MODEL_DIR=""
```

Step 1. Feature creation and tokenization to generate CORE-BEHRT pretraining data
------------------------
Use `main_create_data`: Stores features as dictionaries with list of lists as values and difference concept data streams as keys (concept, segment, age, abspos,...) holding the patient sequences. Tokenizes the features. Use data_pretrain.yaml config.
```bash
python3 main_create_data.py --config_path PATH/TO/data_pretrain.yaml
```
Step 2. Pretrain CORE-BEHRT
------------------------
Use `main_pretrain`: Pre-trains a standard a BEHRT model on the tokenized features. Use pretrain.yaml config. 
```bash
python3 main_pretrain.py --config_path PATH/TO/pretrain.yaml
```
Step 3. CORE-BEHRT model evaluation
------------------------
### Data Preparation for Feature Extraction
Use `main_create_data_downstream`: The input for this should be identical in format to the data used for `main_create_data`. However, users should ensure to remove any data that could constitute temporal leakage (see Appendix B.1 in FoMo paper for information on phenotyping) or any codes that the users would like to remove. Use create_downstreamdata.yaml config. 
```bash
python3 main_create_data_downstream.py --config_path PATH/TO/create_downstreamdata.yaml
```
### Run feature extraction
Use `main_feature_extraction`: Runs the model on outputs of `main_create_data_downstream` and creates a folder of parquet files with the following fields: `PID`, `subject_id`, `prediction_time`, and `features`, where features is a list containing embeddings from the last hidden layer. These outputs can then be used for linear probing. Use linear_prob_features.yaml config. 
```bash
python3 main_create_data_downstream.py --config_path PATH/TO/create_downstreamdata.yaml
```
Note that both scripts need to be run separately for each model evaluation task. 

Step 4. Evaluate using CORE-BERT features
------------------------
To evaluate model performance, we use the following script to train logistic regression classifiers with 5-fold cross-validation using scikit-learn. 
This includes few-shot experiments with varying training set sizes: 100, 1,000, 10,000, and the full training set, evaluated on a fixed test set: 
```bash
sh $PROJECT_ROOT/src/ehr_foundation_model_benchmark/linear_prob/run_linear_prob_with_few_shots.sh \
  --base_dir $CORE_BEhRT_FEATURES_DIR \
  --output_dir $EVALUATION_DIR \
  --meds_dir $OMOP_MEDS \
  --model_name corebehrt
```

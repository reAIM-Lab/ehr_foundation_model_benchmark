# Columbia CEHR-BERT benchmark Pipeline
CEHR-BERT uses the OMOP data as the input directly.

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

Let's set up some environment variables
```bash
export OMOP_DIR=""
export CEHR_BERT_DATA_DIR=""
export CEHR_BERT_MODEL_DIR=""
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
We generate the pretraining data using the following command
```bash
sh src/ehr_foundation_model_benchmark/evaluations/cehrbert/create_cehrbert_pretraining_data.sh \
  --input_folder $OMOP_DIR \
  --output_folde $CEHR_BERT_DATA_DIR \
  --start_date "1985-01-01"
```


# Columbia MEDS Data Processing Pipeline Demo

This guide explains how to process Columbia MEDS data, generate samples, convert the data for `meds_reader`, and train predictive models with `GenHPF`. Follow these steps to install the necessary packages, process the data, and run the model training pipeline.

## Requirements
Ensure you have the following installed:
- Python 3.10
- Conda for environment management
- Required Python packages (specified below)

## Usage

This pipeline takes two command-line arguments:
1. **`INPUT_MEDS`**: Path to the input MEDS data file.
2. **`OUTPUT_MEDS`**: Path to the output directory where processed data will be saved.

To run the pipeline, set up the following environment variables:

```bash
export INPUT_MEDS=$1
export OUTPUT_MEDS_TEMP="$2/temp"
export OUTPUT_MEDS="$2/meds_sample"
export OUTPUT_MEDS_READER="$2/meds_sample_reader"
```

## Steps

### 1. Generate a Sample of Columbia Data
Run the following command to generate a sample of 5,000 records from the Columbia MEDS dataset:

```bash
PYTHONPATH=./:$PYTHONPATH python sample_columbia_meds.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP --sample_size 5000
```

### 2. Combine Unit with Code for Numeric Events
Install the `MEDS_transforms` package, set additional environment variables, and run `MEDS_transform-runner` to combine units with codes for numeric events:

```bash
pip install MEDS_transforms

export COLUMBIA_MEDS_SAMPLE=$OUTPUT_MEDS_TEMP
export COLUMBIA_MEDS_SAMPLE_UNIT_CONCATENATED=$OUTPUT_MEDS

# Run the transformation
MEDS_transform-runner "pipeline_config_fp=transform_columbia_meds_sample.yaml"

# Copy metadata
cp -r $OUTPUT_MEDS_TEMP/metadata $OUTPUT_MEDS
```

### 3. Run `meds_reader`
Convert the sampled data into a format compatible with `meds_reader`:

```bash
pip install meds_reader

# Convert to meds_reader format
meds_reader_convert $OUTPUT_MEDS $OUTPUT_MEDS_READER
```

### 4. Extract Task-Specific Data
Run the extraction task to prepare `OMOP` data related to 30-day readmission rates:

```bash
sh src/MEDS_DEV/helpers/extract_task.sh $OUTPUT_MEDS OMOP readmission/general_hospital/30d
```

### 5. Pre-train FEMR
To pre-train the `FEMR` model, follow the instructions in the FEMR repository: [FEMR OMOP Meds V3 Tutorial](https://github.com/ChaoPang/femr/tree/omop_meds_v3_tutorial/src/femr/omop_meds_tutorial).

### 6. Train GenHPF

Clone the `REMed` repository and set up the environment:

```bash
git clone https://github.com/starmpcc/REMed.git
cd REMed
conda create -n GenHPF python=3.10
conda activate GenHPF

# Install required Python packages
pip install numpy pandas tqdm treelib transformers pyspark polars torch
pip install performer_pytorch recurrent_memory_transformer_pytorch==0.2.2 transformers==4.30.1 accelerate==0.20.3

# Install custom kernel for REMed
cd src/models/kernels/
python setup.py install
```

### 7. Process the Cohort
Set additional environment variables and process the cohort data for `GenHPF`:

```bash
export MEDS_PATH="set this"
export GEN_HPF_PROCESSED_MEDS="set this"
export GEN_HPF_SAVE_DIR="set this"

python scripts/meds/process_meds.py $MEDS_PATH/data \
  --metadata $MEDS_PATH/metadata \
  --cohort $MEDS_PATH/task_labels/readmission/general_hospital/30d/ \
  --output_dir $GEN_HPF_PROCESSED_MEDS --rebase --workers 4
```

### 8. Train the GenHPF Model
Run the following command to train the `GenHPF` model with `accelerate`:

```bash
accelerate launch \
    --config_file config/config.json \
    --num_processes 1 \
    main.py \
    --src_data meds \
    --input_path $GEN_HPF_PROCESSED_MEDS \
    --save_dir $GEN_HPF_SAVE_DIR \
    --pred_targets meds_single_task \
    --train_type short \
    --lr 1e-5 \
    --n_agg_layers 4 \
    --pred_dim 128 \
    --batch_size 16 \
    --max_seq_len 512 \
    --dropout 0.3 \
    --seed 2020 \
    --patience 5
```

This completes the data processing, transformation, and model training pipeline for Columbia MEDS data.
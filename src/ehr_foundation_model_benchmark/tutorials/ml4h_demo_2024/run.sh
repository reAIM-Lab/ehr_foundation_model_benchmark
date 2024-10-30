export INPUT_MEDS=$1
export OUTPUT_MEDS_TEMP="$2/temp"
export OUTPUT_MEDS="$2/meds_sample"
export OUTPUT_MEDS_READER="$2/meds_sample_reader"

# Generate a sample of columbia data
PYTHONPATH=./:$PYTHONPATH python sample_columbia_meds.py --input_meds $INPUT_MEDS --output_meds $OUTPUT_MEDS_TEMP --sample_size 5000

# Combine unit with code for numeric events
pip install MEDS_transforms
export COLUMBIA_MEDS_SAMPLE=$OUTPUT_MEDS_TEMP
export COLUMBIA_MEDS_SAMPLE_UNIT_CONCATENATED=$OUTPUT_MEDS
MEDS_transform-runner "pipeline_config_fp=transform_columbia_meds_sample.yaml"
cp -r $OUTPUT_MEDS_TEMP/metadata $OUTPUT_MEDS

# Run meds_reader
pip install meds_reader
meds_reader_convert $OUTPUT_MEDS $OUTPUT_MEDS_READER

# Run
sh src/MEDS_DEV/helpers/extract_task.sh $OUTPUT_MEDS OMOP readmission/general_hospital/30d

# Pre-train FEMR
# Please following the instructions at https://github.com/ChaoPang/femr/tree/omop_meds_v3_tutorial/src/femr/omop_meds_tutorial

# Train GenHPF
git clone https://github.com/starmpcc/REMed.git
cd REMed
conda create -n GenHPF python=3.10
conda activate GenHPF
pip install numpy pandas tqdm treelib transformers pyspark polars torch
pip install performer_pytorch recurrent_memory_transformer_pytorch==0.2.2 transformers==4.30.1 accelerate==0.20.3
cd src/models/kernels/
python setup.py install

# Process the cohort
export MEDS_PATH="set this"
export GEN_HPF_PROCESSED_MEDS="set this"
export GEN_HPF_SAVE_DIR="set this"
python scripts/meds/process_meds.py $MEDS_PATH/data \
  --metadata $MEDS_PATH/metadata \
  --cohort $MEDS_PATH/task_labels/readmission/general_hospital/30d/ \
  --output_dir $GEN_HPF_PROCESSED_MEDS --rebase --workers 4

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
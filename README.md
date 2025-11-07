## Finetune for Motor Codes

### Setup

```bash
conda create -n femr python=3.10
pip install -e .
pip install meds_evaluation-0.1.dev95+g841c87f-py3-none-any.whl
```

### Revise linear probing codes for finetuning
We execute run_motor.sh inside "src/femr/omop_meds_tutorial/evaluation" for fitting a logistic regression head on pretrained model

```bash
bash run_motor.sh \
  --pretraining_data   /data/models/motor_8k \ 
  --meds_reader        /data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort-reader \
  --num_proc           64 \
  --model_path         /data/models/motor_8k/output \
  --model_name         motor \
  --tokens_per_batch   65536 \
  --device             cuda:0 \
  --min_subjects_per_batch 8 \
  --ontology_path      /data/models/motor_8k/ontology.pkl \
  --main_split_path    /data/models/motor_8k/main_split.csv \
  --output_dir   xxx \
  /data2/processed_datasets/mimic/patient_outcome_tasks/  # path to patient outcome cohort, change to /data2/processed_datasets/mimic/phenotype_task/ for phenotype task
```

run_motor.sh execute three commands

1. "python -u -m femr.omop_meds_tutorial.evaluation.generate_motor_features"

It calls femr.models.architecture.embedding.compute_features to generate batches for task cohort and then generate embeddings. 

2. "python -u -m femr.omop_meds_tutorial.evaluation.finetune_motor"

Note that it is not for finetuning (the name is a bit confusing and we can revise that), but mainly create labels for cohorts (call femr.featurizers.join_labels) and fit logistic regressions

3. "meds-evaluation-cli"

Do bootstrapping for logistic regression prediction results

To do:
We do finetuning for the whole model here, so we need to revise both "generate_motor_features" and "finetune_motor" to tune the weight of both logistic regression head and the pretrained model. The codes for generate batches and bootstrapping can be reused.
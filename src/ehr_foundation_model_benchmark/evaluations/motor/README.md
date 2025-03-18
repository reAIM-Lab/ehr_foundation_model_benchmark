# Columbia MEDS benchmark Pipeline
## MOTOR


```bash
export OMOP_MEDS_READER=""
export ATHENA_DATA=""
export PRETRAINING_DATA=""
export MOTOR_MODEL_DIR=""
```

### Construct the list of codes to remove from the motor pre-training tasks
Generate the extensive list of codes using the OHDSI vocabulary for the codes used in the evaluation tasks. 
For SNOMED codes, we retrieve all the descendants from concept_ancestor first, 
then we retrieve all the non-standard source concepts for those descendant concepts using the "Maps to" relationship from concept_relationship.
```bash
# This should point to codes_to_remove.csv that we constructed 
export DISEASE_CODES_TO_REMOVE="$PRETRAINING_DATA/codes_to_remove.csv"
# The destination file name to extract motor codes to skip the pretraining tasks 
export MOTOR_CODES_TO_SKIP="$PRETRAINING_DATA/motor_codes_to_remove.parquet"
python -u -m ehr_foundation_model_benchmark.evaluations.extract_motor_codes_to_skip_pretraining \
  --athena_path $ATHENA_DATA \
  --disease_codes_to_remove $DISEASE_CODES_TO_REMOVE \
  --motor_codes_to_skip  $MOTOR_CODES_TO_SKIP \
  --meds_reader $OMOP_MEDS_READER
```

Following are the number of codes retrieved from the OHDSI vocabulary as well as the number of codes that will be removed from the MOTOR pretraining tasks
```aiignore
Number of SNOMED codes found in total: 21144
Number of SNOMED codes to skip from motor pretraining: 0
Number of ICD codes found in total: 53567
Number of ICD codes to skip from motor pretraining: 13567
Number of other codes found in total: 24463
Number of other codes to skip from motor pretraining: 0
```

### Benchmark MOTOR with evaluation codes removed
Follow the instructions at [MOTOR on Columbia MEDS](https://github.com/ChaoPang/femr/tree/omop_meds_v3_tutorial/src/femr/omop_meds_tutorial#step-4-preparing-for-pretraining) starting from Step 4. 
For step 4, we do need to add the additional argument `motor_codes_to_skip` to the command to exclude the evaluation codes and their descendants from the pretraining tasks, see the command below. 
From Step 5 onward, you could just follow the steps as-is.  
```bash
# Navigate to the femr folder
python -u -m femr.omop_meds_tutorial.prepare_motor \
  --pretraining_data $PRETRAINING_DATA \
  --athena_path $ATHENA_DATA \
  --meds_reader $OMOP_MEDS_READER \
  --motor_codes_to_skip $MOTOR_CODES_TO_SKIP
```
Verify that the motor tasks do not contain any codes that we want to remove
```python
import os
import pickle
import polars as pl
motor_task_pickle_file = os.path.join(os.environ["PRETRAINING_DATA"], "motor_task.pkl")
motor_codes_to_skip_file = os.environ["MOTOR_CODES_TO_SKIP"]
with open(motor_task_pickle_file, "rb") as f:
    tasks = pickle.load(f)
pretraining_task_codes = [t[0] for t in tasks.pretraining_task_info]
print(pretraining_task_codes[:100])
motor_codes_to_skip = pl.read_parquet(motor_codes_to_skip_file)
# this should return zero rows
print(motor_codes_to_skip.filter(pl.col("code").is_in(pretraining_task_codes)))
```
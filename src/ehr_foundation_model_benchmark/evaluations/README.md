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

```bash
python -u -m femr.omop_meds_tutorial.prepare_motor \
  --pretraining_data $PRETRAINING_DATA \
  --athena_path $ATHENA_DATA \
  --meds_reader $OMOP_MEDS_READER
  --codes_to_skip $MOTOR_CODES_TO_SKIP
```
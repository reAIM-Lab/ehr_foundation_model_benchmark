# Columbia MEDS benchmark Pipeline
## MOTOR


```bash
export OMOP_MEDS_READER=""
export ATHENA_DATA=""
export PRETRAINING_DATA=""
```

```bash
python -u -m femr.omop_meds_tutorial.prepare_motor \
  --pretraining_data $PRETRAINING_DATA \
  --athena_path $ATHENA_DATA \
  --meds_reader $OMOP_MEDS_READER
```
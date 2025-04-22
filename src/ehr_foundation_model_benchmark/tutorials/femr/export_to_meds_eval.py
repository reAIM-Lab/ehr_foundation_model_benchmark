import polars as pl
import pandas as pd
import sys
sys.path.append('/data/mchome/ffp2106/femr/src/femr')
import splits
from pathlib import Path
import numpy as np

from scipy.special import expit


import json

data_json = '/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/tabular/lightgbm_predictions_hf_readmission_meds.json'
data_labels = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/motor/labels/hf_readmission_meds.parquet"


def apply_mask(values, mask):
    def apply(k, v):
        if len(v.shape) == 1:
            return v[mask]
        elif len(v.shape) == 2:
            return v[mask, :]
        else:
            assert False, f"Cannot handle {k} {v.shape}"

    return {k: apply(k, v) for k, v in values.items()}

labels = pd.read_parquet(data_labels)
# labels = labels[labels.subject_id.isin(features["subject_ids"])]
labels = labels.sort_values(["subject_id", "prediction_time"])

# labeled_features = femr.featurizers.join_labels(features, labels)
main_split = splits.SubjectSplit.load_from_csv(str(Path('/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/motor') / 'main_split.csv'))
test_mask = np.isin(labels['subject_id'], main_split.test_subject_ids)

test_data = apply_mask(labels, test_mask)


print(test_data)
print(test_data['subject_id'])
print(test_data['prediction_time'])
print(test_data['boolean_value'])

# exit()


# Step 1: Read the JSON file
# Assume your file is 'data.json' and it's line-delimited (one JSON object per line)

with open(data_json, 'r') as f:
    data = json.load(f)
print(len(data['predictions']))
print(len(data['subject_ids']))

assert len(data['predictions']) == len(test_data['subject_id']), "Mismatch in prediction and label counts."

# Compute sigmoid of predictions
predicted_boolean_probability = expit(np.array(data['predictions']))

# Create a Polars DataFrame
df = pl.DataFrame({
    "subject_id": test_data['subject_id'],
    "prediction_time": test_data['prediction_time'],
    "boolean_value": test_data['boolean_value'],
    "predicted_boolean_probability": predicted_boolean_probability,
})

# Optional: enforce schema explicitly
df = df.with_columns([
    pl.col("subject_id").cast(pl.Int64),
    pl.col("prediction_time").cast(pl.Datetime("us")),
    pl.col("boolean_value").cast(pl.Boolean),
    pl.col("predicted_boolean_probability").cast(pl.Float64),
])

# Save to Parquet
df.write_parquet("output_file.parquet")
import numpy as np
import pandas as pd
from ehr_foundation_model_benchmark.tools.path import files

from ehr_foundation_model_benchmark.tools.mappings import compute_most_common_units

demo = False

def compute_percentage(count, df):
    return count, (count / len(df)) * 100

total_rows_acc = 0
missing_values_acc = 0

val_df = []
val_a = []
val_b = []
val_c = []
val_d = []
val_e = []

most_common_units = compute_most_common_units()

for i, file in enumerate(files):
    # file = file.replace(".snappy.parquet", "-harmonized-harmonized-3.snappy.parquet")
    file = file.replace(".snappy.parquet", "-harmonized-v2.snappy.parquet")
    print()
    print("File", file)
    
    df = pd.read_parquet(file)
    # print(df.columns)
    val_df.append(len(df))

    cdt_n = ((~df['value_as_number'].isnull())) & (df['harmonized_unit_concept_id'].isnull())
    cdt_c = ((~df['value_as_concept_id'].isnull())) & (df['harmonized_unit_concept_id'].isnull())

    cdt = ((~df['value_as_number'].isnull()) | (~df['value_as_concept_id'].isnull())) & (df['harmonized_unit_concept_id'].isnull())
    cdt2 = cdt & (df['unit_concept_id'].isnull()) # is null vs 0?
    print("Non harmonized with value as number", compute_percentage(np.count_nonzero(cdt_n), df))
    print("Non harmonized with value as concept", compute_percentage(np.count_nonzero(cdt_c), df))
    val_a.append(np.count_nonzero(cdt_n))
    val_b.append(np.count_nonzero(cdt_c))
    val_c.append(np.count_nonzero(cdt))
    val_d.append(np.count_nonzero(cdt2))

    print("Non harmonized with value as number or as concept", compute_percentage(np.count_nonzero(cdt), df))
    print("Non harmonized with value as number or as concept AND with NaN unit", compute_percentage(np.count_nonzero(cdt2), df))

    result = df[cdt].groupby('measurement_concept_id').agg(
        unique_unit_count=('unit_concept_id', 'nunique'),
        total_measurements=('measurement_id', 'count')
    ).reset_index()

    result = result.sort_values(by='total_measurements', ascending=False)

    # Display or print the result
    result['most_common_unit_id'] = result['measurement_concept_id'].apply(lambda x: most_common_units[x])
    result['file'] = file
    print(result)
    print(result['unique_unit_count'].value_counts())
    result.to_csv(f'agg-{i}.csv')
    result_cdt = (result['unique_unit_count'] > 1)
    val_e.append(result[result_cdt]['total_measurements'].sum())
    print(compute_percentage(result[result_cdt]['total_measurements'].sum(), df))
    result_cdt = (result['unique_unit_count'] == 1)
    print(compute_percentage(result[result_cdt]['total_measurements'].sum(), df))

    # some measurements can have NA with other units so smaller ones
    result_cdt = (result['unique_unit_count'] == 0) # why it does not match Non harmonized with value as number or as concept AND with NaN unit
    print(compute_percentage(result[result_cdt]['total_measurements'].sum(), df))
    # print(df[['measurement_concept_id', 'unit_concept_name', 'value_as_number', 'harmonized_value_as_number',
    #     'harmonized_unit_concept_id']])

    cdt_null_info = ~df['harmonized_unit_concept_id'].isnull() & \
        df['value_as_number'].isnull() & \
        df['value_as_concept_id'].isnull()
    print("Null values in harmonized units", compute_percentage(np.count_nonzero(cdt_null_info), df))

    result2 = df[~df['harmonized_unit_concept_id'].isnull()].groupby('measurement_concept_id').agg(
        unique_unit_count=('harmonized_unit_concept_id', 'nunique'),
        total_measurements=('measurement_id', 'count')
    ).reset_index()
    result2 = result2.sort_values(by='total_measurements', ascending=False)
    print(result2)
    print(result2['unique_unit_count'].value_counts())
    # no overlap?

    # Count the number of rows in the DataFrame
    total_rows = len(df)
    total_rows_acc += total_rows

    # Count the number of missing values in the specified column
    missing_values = df['harmonized_value_as_number'].isnull().sum()
    missing_values_acc += missing_values

    # Calculate the percentage of missing values
    percentage_missing = (missing_values / total_rows) * 100

    print(f"Total rows: {total_rows}")
    print(f"Non harmonized: {missing_values}")
    print(f"Percentage of Non harmonized: {percentage_missing:.2f}%")

    if demo:
        break

import pickle
with open('stats.pkl', 'wb') as f:
    pickle.dump((val_df, val_a, val_b, val_c, val_d, val_e), f)


percentage_missing_acc = (missing_values_acc / total_rows_acc) * 100

print(f"Total rows: {total_rows_acc}")
print(f"Missing values: {missing_values_acc}")
print(f"Percentage of missing values: {percentage_missing_acc:.2f}%")
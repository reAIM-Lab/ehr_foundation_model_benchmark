from pathlib import Path
from ehr_foundation_model_benchmark.tools.path import files
import json
import pandas as pd

base_dir = Path(files[0]).parent

files_info = sorted(list(base_dir.glob("*-processed_stats.csv")))
print(list(files_info))

all_data = []
total_rows = 0
for i, file_info in enumerate(files_info):
    with open(file_info, 'r') as f:
        data = json.load(f)
    if i == 0:
        data = [[*d1, 0] for d1 in data]
    local_row_file = int(data[0][1] / data[0][2] * 100)
    local_row_defined_file = int(data[2][1] / data[2][2] * 100)
    print(data[2][1], data[2][2])
    data = [[i, *d1, local_row_file, local_row_defined_file] for d1 in data]
    all_data.extend(data)
    print(data[1])
    total_rows += int(data[0][2] / data[0][3] * 100)
    # total_rows += int(data[4][2] / data[4][3] * 100)

print(all_data)

df = pd.DataFrame(all_data, columns=['file', 'step', 'n_rows', 'local_percentage', 'time_s', 'n_total_rows_local_file', 'n_total_rows_local_file_defined_meas'])
df = df[~(df['step'] == 'Undefined labs')]
df = df[df['n_rows'] > 0]
df.to_csv('stats_agg.csv')
# df.drop(0, inplace=True)
df.dropna()

df2 = df.groupby('step', as_index=False).sum()
# df2['local_percentage'] = df2['n_rows'] / df2['n_total_rows_local_file'] * 100
df2['local_percentage'] = df2['n_rows'] / df2['n_total_rows_local_file_defined_meas'] * 100
print(df2)

df = df[df['step'].isin(['Single unit labs', 'Converted rows', 'Copied rows'])]
print(df)
# df['total_rows'] = (df['n_rows'] / df['local_percentage'] * 100).astype(int)
# total_rows = df['total_rows'].sum()

grouped_df = df.groupby('step', as_index=False)['n_rows'].sum()
grouped_df['total_percentage'] = (grouped_df['n_rows'] / total_rows) * 100

print()
print(grouped_df)

print()
print(grouped_df.sum())

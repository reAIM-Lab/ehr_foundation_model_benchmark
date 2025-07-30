import pandas as pd
import os

df_stats_table = pd.DataFrame(columns=['cohort_name', 'split_dataset', 'perc'])

COHORT_LIST = ['AMI', 'Pancreatic Cancer', 'Celiac', 'SLE', 'HTN', 'MASLD', 'CLL', 'Osteoporosis', 'Ischemic Stroke', 'T2DM', 'Schizophrenia']

for cohort_name in COHORT_LIST:
    cohort_name = cohort_name.replace(' ', '_')
    print(f"\n=== {cohort_name} ===")
    datasets = ['train', 'tuning', 'held_out']
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        df = pd.read_parquet(os.path.join(cohort_name, f'{dataset}.parquet'))
        df_stats = df.groupby('boolean_value').count().reset_index()
        df_stats['perc'] = df_stats['subject_id'] / df_stats['subject_id'].sum()
        df_stats_table.loc[len(df_stats_table)] = [cohort_name, dataset, df_stats['perc'].values[1]]
print(df_stats_table)
#df_stats_table.to_csv('cohort_prevalence.csv', index=False)
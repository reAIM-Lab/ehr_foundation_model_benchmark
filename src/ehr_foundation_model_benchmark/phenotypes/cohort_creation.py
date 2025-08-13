import pandas as pd
from sqlalchemy import create_engine
from jinja2 import Template
import os
import urllib.parse

FILE_PATH = '/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform'
SAMPLE_PERCENTAGE = 0.1

driver_name = "ODBC Driver 17 for SQL Server"
server = ""
username = ""
password = ""
database_name = 'ohdsi_cumc_2023q4r3_deid'
conn_url = f"DRIVER={driver_name};SERVER={server};UID={username};PWD={password};DATABASE={database_name}"
quoted_conn = urllib.parse.quote_plus(conn_url)
engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={quoted_conn}",
    fast_executemany=True  # This improves performance for bulk inserts
)

MIN_OBS_YEARS = 2
PREDICTION_YEARS = 1
COHORT_LIST = ['AMI', 'Pancreatic Cancer', 'Celiac', 'SLE', 'HTN', 'MASLD', 'CLL', 'Osteoporosis', 'Ischemic Stroke', 'T2DM', 'Schizophrenia']

def get_split_data(file):
    """
    Get the patient split data which contains subject_id, split columns.
    :param file:
    :return: pandas dataframe
    """
    split_data = pd.read_parquet(file)
    return split_data

def render_query(query, **kwargs):
    with open(query) as f:
        query = f.read()
        template = Template(query)
        rendered_query = template.render(**kwargs)
        return rendered_query

def get_sample_data(cohort_name, percentage, case_control_query):
    context = {'database_name': database_name,
                'cohort_name': cohort_name}
    case_control_query = render_query(case_control_query, **context)
    df = pd.read_sql(case_control_query, engine)
    df = df.sample(frac=percentage)
    return df

def save_sample_data_to_db(df, cohort_name):
    df.to_sql(schema='results', name = 'phenotype_temp_sample', con=engine, if_exists='replace', index=False)
    return f'Temp table created for {cohort_name}'


def get_visit_level_data(visit_query, sample_data, min_obs_years, prediction_years, cohort_name):
    context = {'database_name': database_name,
                'cohort_name': cohort_name,
                'min_obs_years': min_obs_years,
                'prediction_years': prediction_years}
    visit_query = render_query(visit_query, **context)
    visit_level_data = pd.read_sql(visit_query, engine)
    sample_visit_level_data = sample_data[['subject_id']].merge(visit_level_data, how='inner', on='subject_id')
    return sample_visit_level_data

def print_stats(df_stats):
    total = df_stats['subject_id'].sum()
    print(f"  Total subjects: {total:,}")
    for _, row in df_stats.iterrows():
        count = row['subject_id']
        percentage = row['perc'] * 100
        value = row['subject_cohort']
        print(f"  {value}: {count:,} ({percentage:.1f}%)")

def generate_cohort(visit_level_data, cohort_name, split_data, sampled_df):
    merged_data = sampled_df.merge(split_data, how='inner', on='subject_id') \
        .merge(visit_level_data, how='inner', on='subject_id')

    ### Print out cohort prevalence in each split
    print(f"\n=== {cohort_name} ===")
    datasets = ['train', 'tuning', 'held_out']
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        df_stats = merged_data[merged_data['split'] == dataset]
        df_stats = df_stats[['subject_id', 'subject_cohort']].drop_duplicates()
        df_stats = df_stats.groupby('subject_cohort').count().reset_index()
        df_stats['perc'] = df_stats['subject_id'] / df_stats['subject_id'].sum()
        print_stats(df_stats)

    train_data = merged_data[merged_data['split'] == 'train'].drop(columns=['split'])
    tuning_data = merged_data[merged_data['split'] == 'tuning'].drop(columns=['split'])
    held_out_data = merged_data[merged_data['split'] == 'held_out'].drop(columns=['split'])

    cohort_path = os.path.join(FILE_PATH, f'phenotype_cohorts_min_obs_{MIN_OBS_YEARS}_years', cohort_name.replace(' ', '_'))
    os.makedirs(cohort_path, exist_ok=True)

    train_data.to_parquet(f"{os.path.join(cohort_path, 'train.parquet')}")
    tuning_data.to_parquet(f"{os.path.join(cohort_path, 'tuning.parquet')}")
    held_out_data.to_parquet(f"{os.path.join(cohort_path, 'held_out.parquet')}")

def main():
    split_patient = get_split_data(os.path.join(FILE_PATH, 'metadata/subject_splits.parquet'))
    for cohort_name in COHORT_LIST:
        print(f'Processing cohort {cohort_name}')
        sample_data = get_sample_data(cohort_name=cohort_name, percentage=SAMPLE_PERCENTAGE, case_control_query='cohort_pos_neg_query/case_control_query.sql')
        print('Save data to the temp table')
        save_sample_data_to_db(sample_data, cohort_name)
        print('Get visit level data')
        visit_level_data = get_visit_level_data('cohort_pos_neg_query/cohort_pos_neg_query.sql',
                                                sample_data = sample_data,
                                                min_obs_years = MIN_OBS_YEARS,
                                                prediction_years = PREDICTION_YEARS,
                                                cohort_name=cohort_name)
        generate_cohort(visit_level_data, cohort_name, split_patient, sample_data)

if __name__ == '__main__':
    main()
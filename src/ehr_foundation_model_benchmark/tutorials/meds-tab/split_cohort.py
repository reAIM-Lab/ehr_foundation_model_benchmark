import pandas as pd
import glob
import os

# main_split = pd.read_csv('/home/ffp2106@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/meds-tab/main_split.csv')
# print(main_split)
# print(main_split['split_name'].unique())

main_split = pd.read_parquet('/data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort/metadata/subject_splits.parquet', engine='fastparquet')

# train_subjects = main_split['subject_id'][main_split['split_name'] == 'train']
# test_subjects = main_split['subject_id'][main_split['split_name'] == 'test']
# test_subjects = main_split['subject_id'][main_split['split_name'] == 'held_out']

train_subjects = main_split['subject_id'][main_split['split'] == 'train']
# test_subjects = main_split['subject_id'][main_split['split_name'] == 'test']
test_subjects = main_split['subject_id'][main_split['split'] == 'held_out']
tuning_subjects = main_split['subject_id'][main_split['split'] == 'tuning']

# # sample 10% of subjects
# train_subjects = main_split.loc[main_split['split_name'] == 'train', 'subject_id']
# test_subjects = main_split.loc[main_split['split_name'] == 'test', 'subject_id']

# # Compute 10% of total subjects (train + test)
# total_subjects = len(train_subjects) + len(test_subjects)
# sample_size = int(0.1 * total_subjects)

# # Sample from train subjects to create tuning dataset
# tuning_subjects = train_subjects.sample(n=sample_size, random_state=42)

# main_split.loc[
#     main_split['subject_id'].isin(tuning_subjects), 'split_name'
# ] = 'tuning'

# # Check the updated splits
# print(main_split['split_name'].value_counts())
# print(main_split.head())

# # Save updated version if needed
# main_split.to_csv(
#     '/home/ffp2106@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/meds-tab/main_split_updated.csv',
#     index=False
# )
# exit()



cohorts = glob.glob('/data/processed_datasets/processed_datasets/mimic/phenotype_task_binary/*/*_cohort.parquet')
output_path = "/data/processed_datasets/processed_datasets/mimic/phenotype_task_split/"
for cohort in cohorts:
    name = os.path.basename(os.path.dirname(cohort))
    print(name)
    df = pd.read_parquet(cohort, engine='fastparquet')
    df = df[['subject_id','prediction_time','binary_label']]
    df.columns = ['subject_id','prediction_time','boolean_value']
    # print(df)
    df_train = df[df['subject_id'].isin(train_subjects)]
    df_tuning = df[df['subject_id'].isin(tuning_subjects)]
    df_test = df[df['subject_id'].isin(test_subjects)]
    print(df_train)
    print(df_tuning)
    print(df_test)

    op = output_path + name
    os.makedirs(op, exist_ok=True)
    df_train.to_parquet(op + '/train.parquet')
    df_tuning.to_parquet(op + '/tuning.parquet')
    df_test.to_parquet(op + '/held_out.parquet')



    # also change colujmn label name
    # exit()

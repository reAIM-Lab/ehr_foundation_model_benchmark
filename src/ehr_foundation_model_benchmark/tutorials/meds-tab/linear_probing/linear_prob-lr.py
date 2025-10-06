import subprocess


tasks = ["in_hospital_mortality"]

# tasks = ["long_los", "readmission", 'masld', 'celiac', 'stroke']
# tasks = ['Osteoporosis']
# tasks = ['Schizophrenia']

base_command = [
    "python", "/home/ffp2106@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/meds-tab/linear_probing/linear_prob.py",
    # "--features_label_input_dir", "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-fix2-large-katara/{task}_final/tabularize_export",

    # "--features_label_input_dir", "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-fix2-large/{task}_final/tabularize_export",
    "--features_label_input_dir", "/data/processed_datasets/processed_datasets/ehr_foundation_data/outputs/models/meds_tab/output_mimic_v2/{task}_final/tabularize_export/",
    # "--meds_dir", "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/post_transform",
    "--meds_dir", "/data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort/",
    # "--output_dir", "/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/results_probing",
    # "--output_dir", "/data/processed_datasets/processed_datasets/ehr_foundation_data/outputs/models/meds_tab/output",
    "--output_dir", "/data/processed_datasets/processed_datasets/ehr_foundation_data/outputs/katara-v2",
    "--model_name", "medstab-lr",
    "--task_name", "{task}"
]

for task in tasks:
    print(f"Running task: {task}")
    command = [arg.format(task=task) for arg in base_command]
    subprocess.run(" ".join(command), shell=True, check=True)
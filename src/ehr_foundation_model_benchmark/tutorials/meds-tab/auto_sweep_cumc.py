import os
import subprocess
import sys
import shutil

# -----------------------------------------------------------------------------
# CONFIGURATION: adjust these if your base paths differ
# -----------------------------------------------------------------------------
BASE_ROOT = (
    "/data2/processed_datasets"
    "/ehr_foundation_data/ohdsi_cumc_deid/"
    "ohdsi_cumc_deid_2023q4r3_v3_mapped"
    # "/data/processed_datasets/processed_datasets/ehr_foundation_data/outputs"
)
EXTRACT_SCRIPT = os.path.expanduser(
    "/home/ffp2106@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/"
    "src/ehr_foundation_model_benchmark/"
    "tutorials/meds-tab/linear_probing/extract_features.py"
)
BENCHMARK_SCRIPT = os.path.expanduser(
    "/home/ffp2106@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/"
    "src/ehr_foundation_model_benchmark/"
    "tutorials/meds-tab/linear_probing/linear_prob-xgboost.py"
)
RESHARD_SCRIPT = "/home/ffp2106@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/meds-tab/reshard.py"
SELECT_FEATURES_SCRIPT = "/home/ffp2106@mc.cumc.columbia.edu/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/meds-tab/select_features.py"
MEDS_TAB_XGBOOST_CMD = "meds-tab-xgboost"

POST_TRANSFORM_DIR = os.path.join(BASE_ROOT, "post_transform")
# POST_TRANSFORM_DIR = "/data/raw_data/mimic/files/mimiciv/meds_v0.6/3.1/MEDS_cohort/"
# OUTPUT_FIX2_LARGE = os.path.join(BASE_ROOT, "models/meds_tab/output_mimic_v2")
OUTPUT_FIX2_LARGE = os.path.join(BASE_ROOT, "models/meds_tab/output-2year")

OUTPUT_FIX2_LARGE_XGB = os.path.join(BASE_ROOT, "models/meds_tab/output-2year-xgb")

# TASKS = ["long_los", "death", "readmission"]
# TASKS = ["CLL"]
# TASKS = ["AMI"]
# 'in_hospital_mortality', 
# TASKS = ['in_hospital_mortality']
# TASKS = ['HTN', 'Ischemic_Stroke', 'MASLD', 'Osteoporosis', 'Pancreatic_Cancer', 'Schizophrenia',  'SLE', 'T2DM']
TASKS = ['CLL', 'Celiac', 'SLE', 'T2DM']
# TASKS = ['long_los', 'readmission', 'celiac', 'masld', 'stroke']
# TASKS = ['Osteoporosis']
# TASKS = ['Schizophrenia']
SAMPLE_SIZES = [100, 1000, 10000, 100000] #[10000, 1000, 100]
# SAMPLE_SIZES = [100, 1000, 10000]

# -----------------------------------------------------------------------------
# HELPER to run a command and abort on failure
# -----------------------------------------------------------------------------
def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

labels_folder_expanded = OUTPUT_FIX2_LARGE.replace('outputs', 'XXX').replace("output", "labels").replace('XXX', 'outputs')
def main():
    perfs = []
    for task in TASKS:
        # Step 1: linear probing via XGBoost
        features_label_input = os.path.join(
            OUTPUT_FIX2_LARGE, f"{task}_final", "tabularize_export"
        )
        label_input_raw = os.path.join(
            # labels_folder_expanded, f"{task}"
            # OUTPUT_FIX2_LARGE, f"{task}"
            labels_folder_expanded, f"{task}"
        )
        features_label_input_raw = os.path.join(
            OUTPUT_FIX2_LARGE, f"{task}_final", "tabularize"
        )
        probing_output = os.path.join(OUTPUT_FIX2_LARGE, f"{task}_probing_xgb_updated-v2")
        os.makedirs(probing_output, exist_ok=True)

        # if True:
        if True:#not os.path.exists(features_label_input):
            run([
                sys.executable, EXTRACT_SCRIPT,
                task, OUTPUT_FIX2_LARGE, labels_folder_expanded

            ])
        else:
            print("Skipping extracting features")
        # exit()

        run([
            sys.executable, BENCHMARK_SCRIPT,
            "--features_label_input_dir", features_label_input,
            "--meds_dir", POST_TRANSFORM_DIR,
            "--output_dir", probing_output,
            "--model_name", "medstab",
            "--task_name", task
        ])
        # exit()

        # Steps 2–4: for each sample size
        for n in SAMPLE_SIZES:
            # if task == "death" and n == 100:
            #     print("Skipping death task for sample size 100")
            #     continue
            cohort_dir = os.path.join(probing_output, task)

            # Step 2: reshard into labels_{n}
            run([
                sys.executable, RESHARD_SCRIPT,
                "--meds_data", os.path.join(POST_TRANSFORM_DIR, "data"),
                "--cohort_input", os.path.join(cohort_dir, f"medstab_{n}.parquet"),
                "--cohort_output", os.path.join(cohort_dir, f"labels_{n}"),
                "--split", "all"
            ])

            # Step 3: select top‐n features
            run([
                sys.executable, SELECT_FEATURES_SCRIPT,
                str(n), task, cohort_dir, features_label_input_raw, label_input_raw
            ])

            # Step 4: hyperparameter sweep with meds-tab-xgboost
            run([
                MEDS_TAB_XGBOOST_CMD,
                "--multirun",
                "worker=range(0,16)",
                f"input_dir={POST_TRANSFORM_DIR}",
                f"output_dir={OUTPUT_FIX2_LARGE_XGB}",
                f"task_name={task}",
                f"output_model_dir={os.path.join(OUTPUT_FIX2_LARGE_XGB, task)}",
                "do_overwrite=False",
                # "hydra.sweeper.n_trials=2",
                "hydra.sweeper.n_jobs=16",
                "hydra.sweeper.n_trials=100",
                # "hydra.sweeper.n_jobs=16",
                # "tabularization.min_code_inclusion_count=0",
                "tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]",
                # "tabularization.window_sizes=[1d,7d,30d,60d,365d,full]",
                "tabularization.window_sizes=[1d,7d,30d,60d,365d,730d]",
                f"input_label_cache_dir={os.path.join(cohort_dir, f'labels_{n}')}",
                f"input_tabularized_cache_dir={os.path.join(cohort_dir, f'output_{n}')}",
                f"tabularization.filtered_code_metadata_fp={os.path.join(OUTPUT_FIX2_LARGE, f'{task}_final/metadata/codes.parquet')}"
            ])

            # copy output prediction to folder for katara
            # output_to_copy = f'/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/results_probing/{task}'
            output_to_copy = f"/data2/ehr_foundation_data/outputs/katara-v3/{task}"
            xgb_results = f'{OUTPUT_FIX2_LARGE_XGB}/{task}'
            # find last 
            folders = sorted([f for f in os.listdir(xgb_results) if os.path.isdir(os.path.join(xgb_results, f))])
            last_folder = folders[-1] if folders else None

            file_to_copy = f"{xgb_results}/{last_folder}/best_trial/held_out_predictions.parquet"
            os.makedirs(output_to_copy, exist_ok=True)
            shutil.copy2(file_to_copy, f'{output_to_copy}/medstab_{n}.parquet')

            file_perf = f"{xgb_results}/{last_folder}/best_trial/performance.log"
            with open(file_perf, 'r') as f:
                perfs.append((task, n, f.read()))
            
    


            # input()
        # exit()
    print(perfs)

if __name__ == "__main__":
    main()
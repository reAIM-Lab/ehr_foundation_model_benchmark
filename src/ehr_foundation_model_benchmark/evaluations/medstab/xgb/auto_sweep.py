"""
Code to run xgboost models on 100, 1000, and 10000 samples
"""
import os
import subprocess
import sys
import shutil

# -----------------------------------------------------------------------------
# CONFIGURATION: adjust these if your base paths differ
# -----------------------------------------------------------------------------
BASE_ROOT = (
    "XXX"
)
EXTRACT_SCRIPT = os.path.expanduser(
    "postprocessing/extract_features.py"
)
BENCHMARK_SCRIPT = os.path.expanduser(
    "xgb/linear_prob-xgboost.py"
)
RESHARD_SCRIPT = "preprocessing/reshard.py"
SELECT_FEATURES_SCRIPT = "xgb/select_features.py"
MEDS_TAB_XGBOOST_CMD = "meds-tab-xgboost"

POST_TRANSFORM_DIR = os.path.join(BASE_ROOT, "XXX")
OUTPUT_FIX2_LARGE = os.path.join(BASE_ROOT, "XX4")
OUTPUT_FIX2_LARGE_XGB = os.path.join(BASE_ROOT, "XX6")

TASKS = ["long_los", "death", "readmission", "CLL", "AMI"] # edit this line
SAMPLE_SIZES = [100, 1000, 10000]

# -----------------------------------------------------------------------------
# HELPER to run a command and abort on failure
# -----------------------------------------------------------------------------
def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    perfs = []
    for task in TASKS:
        # Step 1: linear probing via XGBoost
        features_label_input = os.path.join(
            OUTPUT_FIX2_LARGE, f"{task}_final", "tabularize_export"
        )
        label_input_raw = os.path.join(
            OUTPUT_FIX2_LARGE.replace("output", "labels"), f"{task}"
        )
        features_label_input_raw = os.path.join(
            OUTPUT_FIX2_LARGE, f"{task}_final", "tabularize"
        )
        probing_output = os.path.join(OUTPUT_FIX2_LARGE, f"{task}_probing_xgb_updated")
        os.makedirs(probing_output, exist_ok=True)

        if not os.path.exists(features_label_input):
            run([
                sys.executable, EXTRACT_SCRIPT,
                task, OUTPUT_FIX2_LARGE, OUTPUT_FIX2_LARGE.replace("output", "labels")

            ])
        else:
            print("Skipping extracting features")

        run([
            sys.executable, BENCHMARK_SCRIPT,
            "--features_label_input_dir", features_label_input,
            "--meds_dir", POST_TRANSFORM_DIR,
            "--output_dir", probing_output,
            "--model_name", "medstab",
            "--task_name", task
        ])

        # Steps 2–4: for each sample size
        for n in SAMPLE_SIZES:
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
                "tabularization.window_sizes=[1d,7d,30d,60d,365d,full]",
                f"input_label_cache_dir={os.path.join(cohort_dir, f'labels_{n}')}",
                f"input_tabularized_cache_dir={os.path.join(cohort_dir, f'output_{n}')}",
                f"tabularization.filtered_code_metadata_fp={os.path.join(OUTPUT_FIX2_LARGE, f'{task}_final/metadata/codes.parquet')}"
            ])

            # copy
            output_to_copy = f'XX7/{task}'
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
            
    print(perfs)

if __name__ == "__main__":
    main()
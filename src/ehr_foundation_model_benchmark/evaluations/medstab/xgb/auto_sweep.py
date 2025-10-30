"""
Code to run xgboost models on 100, 1000, and 10000 samples
"""
import glob
import os
import re
import subprocess
import sys
import shutil
import argparse

# -----------------------------------------------------------------------------
# CONFIGURATION: adjust these if your base paths differ
# -----------------------------------------------------------------------------

EXTRACT_SCRIPT = os.path.expanduser(
    "postprocessing/extract_features.py"
)
BENCHMARK_SCRIPT = os.path.expanduser(
    "xgb/linear_prob-xgboost.py"
)
RESHARD_SCRIPT = "preprocessing/reshard.py"
SELECT_FEATURES_SCRIPT = "xgb/select_features.py"
MEDS_TAB_XGBOOST_CMD = "meds-tab-xgboost"

def get_args():
    parser = argparse.ArgumentParser(
        description="Configuration for featurization"
    )
    parser.add_argument(
        "--phenotypes-dir", 
        type=str, 
        default="XX2",
        help="Path to phenotypes directory."
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="XX1",
        help="Path to data directory."
    )
    parser.add_argument(
        "--output-features-dir", 
        type=str, 
        default="XX1",
        help="Path to features output directory."
    )
    parser.add_argument(
        "--output_few_shots_dir", 
        type=str, 
        default="XX1",
        help="Path to few shots output directory."
    )
    parser.add_argument(
        "--output-reshard-dir", 
        type=str, 
        default="XX1",
        help="Path to reshard output directory."
    )
    parser.add_argument(
        "--output-model-dir", 
        type=str, 
        default="XX1",
        help="Path to model output directory."
    )
    parser.add_argument(
        "--output-results-dir", 
        type=str, 
        default="XX1",
        help="Path to results output directory."
    )
    parser.add_argument(
        "--windows", 
        type=str, 
        default="XX1",
        help="Feature windows to compute."
    )
    parser.add_argument(
        "--tasks", 
        type=str, 
        default="XX1",
        help="Tasks to do"
    )

    args = parser.parse_args()
    args.tasks = args.tasks.split(',')
    return args

args = get_args()

TASKS = args.tasks

# TASKS = ["long_los", "death", "readmission", "CLL", "AMI"] # edit this line
# SAMPLE_SIZES = [100, 1000, 10000] # full = number to make sure

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
        features_input = os.path.join(
            args.output_features_dir, f"{task}_final", "tabularize_export"
        )
        codes_input = os.path.join(
            args.output_features_dir, f"{task}_final", "metadata"
        )
        label_input_raw = os.path.join(
            args.output_reshard_dir, f"{task}"
        )
        features_label_input_raw = os.path.join(
            args.output_features_dir, f"{task}_final", "tabularize"
        )
        # label_input_output = os.path.join(
        #     args.output_labels_dir, f"{task}"
        # )
        probing_output = os.path.join(args.output_model_dir, f"{task}_probing_xgb_updated")
        os.makedirs(probing_output, exist_ok=True)

        if not os.path.exists(features_input):
            run([
                sys.executable, EXTRACT_SCRIPT,
                task, features_label_input_raw, codes_input, label_input_raw, features_input
            ])
        else:
            print("Skipping extracting features")

        run([
            sys.executable, BENCHMARK_SCRIPT,
            "--features_label_input_dir", features_input,
            "--meds_dir", args.data_dir,
            "--output_dir", probing_output,
            "--model_name", "medstab",
            "--task_name", task
        ])

        # Steps 2–4: for each sample size
        cohort_dir = os.path.join(probing_output, task)
        pattern = os.path.join(cohort_dir, "medstab_*.parquet")
        files = glob.glob(pattern)

        # Extract numeric part and pair it with filenames
        file_number_pairs = []
        for f in files:
            match = re.search(r"medstab_(\d+)\.parquet$", os.path.basename(f))
            if match:
                num = int(match.group(1))
                file_number_pairs.append((num, f))

        # Sort by number
        file_number_pairs.sort(key=lambda x: x[0])

        # Separate lists of sorted files and numbers
        sorted_files = [f for _, f in file_number_pairs]
        numbers = [n for n, _ in file_number_pairs]

        for n, file in zip(sorted_files, numbers):
            cohort_dir_labels = os.path.join(cohort_dir, f'labels_{n}')
            cohort_dir_outputs = os.path.join(cohort_dir, f'output_{n}')

            # Step 2: reshard into labels_{n}
            run([
                sys.executable, RESHARD_SCRIPT,
                "--meds_data", os.path.join(args.data_dir, "data"),
                "--cohort_input", os.path.join(cohort_dir, f"medstab_{n}.parquet"),
                "--cohort_output", os.path.join(cohort_dir, f"labels_{n}"),
                "--split", "all"
            ])

            # Step 3: select top‐n features
            run([
                sys.executable, SELECT_FEATURES_SCRIPT,
                str(n), task, cohort_dir, features_label_input_raw, label_input_raw,
                cohort_dir_labels, cohort_dir_outputs
            ])

            # Step 4: hyperparameter sweep with meds-tab-xgboost
            run([
                MEDS_TAB_XGBOOST_CMD,
                "--multirun",
                "worker=range(0,16)",
                f"input_dir={args.data_dir}",
                f"output_dir={args.output_few_shots_dir}",
                f"task_name={task}",
                f"output_model_dir={os.path.join(args.output_few_shots_dir, task)}",
                "do_overwrite=False",
                # "hydra.sweeper.n_trials=2",
                "hydra.sweeper.n_jobs=16",
                "hydra.sweeper.n_trials=100",
                # "hydra.sweeper.n_jobs=16",
                # "tabularization.min_code_inclusion_count=0",
                "tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]",
                f"tabularization.window_sizes=[{args.windows}]",
                f"input_label_cache_dir={cohort_dir_labels}",
                f"input_tabularized_cache_dir={cohort_dir_outputs}",
                f"tabularization.filtered_code_metadata_fp={os.path.join(args.output_features_dir, f'{task}_final/metadata/codes.parquet')}"
            ])

            # copy
            output_to_copy = f'{args.output_results_dir}/{task}'
            xgb_results = f'{args.output_few_shots_dir}/{task}'
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
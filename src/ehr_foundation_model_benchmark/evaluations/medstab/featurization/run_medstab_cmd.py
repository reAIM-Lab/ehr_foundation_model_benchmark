"""
Code to run the featurization for tasks
"""
import subprocess
import traceback
import time
from datetime import datetime
from tqdm import tqdm
import glob
import os
import argparse
import shutil

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
        "--output-reshard-dir", 
        type=str, 
        default="XX1",
        help="Path to model output directory."
    )
    parser.add_argument(
        "--windows", 
        type=str, 
        default="XX1",
        help="Feature windows to compute."
    )

    args = parser.parse_args()
    # print(args.windows)
    # exit()
    return args

args = get_args()

# Constants
N_PARALLEL_WORKERS = 32
SPLITS = ["train", "tuning", "held_out"]
LOG_FILE = "processing_errors.log"
TIME_LOG_FILE = "task_training_times.log"  # File to save training times

def log_error(task, step, exception):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{datetime.now()}] ERROR in Task: {task} | Step: {step}\n")
        f.write(f"{traceback.format_exc()}\n")

# Function to log training times
def log_training_time(task, step, duration):
    with open(TIME_LOG_FILE, "a") as f:
        f.write(f"[{datetime.now()}] Task: {task} | Step: {step} | Duration: {duration:.2f} seconds\n")

def clean_cache(task):
    path = f'{args.output_features_dir}/{task}_final/tabularize'

    files = glob.glob(os.path.join(path, '**', '.*.npz_cache'), recursive=True)
    print(path)
    k = 0
    for file in files:
        try:
            # remove only if name.npz does not exist when .name.npz_cache exists
            base_name = os.path.splitext(file)[0].replace('/.', '/') 
            if os.path.exists(base_name + '.npz'):
                continue
            k += 1
            shutil.rmtree(file)  # Remove the directory containing the cache file
            print(f"Removed cache file: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
            exit()
    print(f"Total cache files removed: {k}")

# Get all phenotype task subfolders
phenotype_tasks = list(sorted([
    name for name in os.listdir(args.phenotypes_dir)
    if os.path.isdir(os.path.join(args.phenotypes_dir, name))
]))

for i, task in (pbar := tqdm(enumerate(phenotype_tasks), total=len(phenotype_tasks))):
    if "logs" in task:
        continue
    
    task = os.path.basename(task)
    task_path = os.path.join(args.phenotypes_dir, task)
    pbar.set_description(task)

    # Step 1: Run reshard.py for each split
    for split in SPLITS:
        print("Resharding", split)

        if task_path.endswith(".parquet"):
            task_path = os.path.dirname(task_path)
            local_split = task.replace(".parquet", "")
        else:
            local_split = split

        cohort_input = os.path.join(task_path, f"{local_split}.parquet")
        cohort_output = os.path.join(args.output_reshard_dir, task)
        if not os.path.exists(cohort_input):
            print(f"Warning: {cohort_input} not found. Skipping.")
            continue

        cmd_reshard = [
            "python",
            "preprocessing/reshard.py",
            "--cohort_input", cohort_input,
            "--meds_data", os.path.join(args.data_dir, "data"),
            "--cohort_output", cohort_output,
            "--split", split
        ]

        start_time = time.time()  # Start timing
        try:
            print("Running:", " ".join(cmd_reshard))
            # subprocess.run(cmd_reshard, check=True)
        except Exception as e:
            log_error(task, f"reshard.py ({split})", e)
        finally:
            duration = time.time() - start_time  # Calculate duration
            log_training_time(task, f"reshard.py ({split})", duration)

    # Step 2: Run meds-tab-describe
    describe_cmd = [
        "meds-tab-describe",
        f"input_dir={os.path.join(args.data_dir, 'data')}",
        f"output_dir={os.path.join(args.output_features_dir, task + '_final')}"
    ]
    start_time = time.time()  # Start timing
    try:
        print("Running:", " ".join(describe_cmd))
        subprocess.run(describe_cmd, check=True)
    except Exception as e:
        print(e)
        import traceback
        print(traceback.format_exc())
        log_error(task, "meds-tab-describe", e)
    finally:
        duration = time.time() - start_time  # Calculate duration
        log_training_time(task, "meds-tab-describe", duration)

    # exit(1)

    # Step 3: Run meds-tab-tabularize-time-series
    tabularize_cmd = [
        "meds-tab-tabularize-time-series",
        "--multirun",
        f"worker=range(0,{N_PARALLEL_WORKERS})",
        "hydra/launcher=joblib",
        f"input_dir={os.path.join(args.data_dir, 'data')}",
        f"output_dir={os.path.join(args.output_features_dir, task + '_final')}",
        "do_overwrite=False",
        f"input_label_dir={os.path.join(args.output_reshard_dir, task)}",
        "tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]",
        f"tabularization.window_sizes=[{args.windows}]"
        # "tabularization.aggs=[code/count]",
        # "tabularization.window_sizes=[full]"
    ]
    start_time = time.time()  # Start timing
    complete = False
    repet = 0
    while not complete:
        crash = False
        try:
            print("Running:", " ".join(tabularize_cmd))
            subprocess.run(tabularize_cmd, check=True)
            complete = True
        except Exception as e:
            log_error(task, "meds-tab-tabularize-time-series", e)
            crash = True
        finally:
            duration = time.time() - start_time  # Calculate duration
            log_training_time(task, "meds-tab-tabularize-time-series", duration)
        if crash:
            print("Crash in meds-tab-tabularize-time-series")
            clean_cache(task)
            repet += 1
            if repet > 5:
                print("Too many crashes.", repet)

  


print("Done")
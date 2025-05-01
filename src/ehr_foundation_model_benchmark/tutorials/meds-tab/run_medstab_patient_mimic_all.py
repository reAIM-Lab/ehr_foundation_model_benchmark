import os
import subprocess
import traceback
import time  # Add this import
from datetime import datetime
from tqdm import tqdm
import glob

# Constants
BASE_PATH = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/"
# PHENOTYPE_PATH = os.path.join(BASE_PATH, "task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years")
PHENOTYPE_PATH = os.path.join(BASE_PATH, "ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated_output/")
# PHENOTYPE_PATH = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/motor/labels"
REDSHARD_DIR = os.path.join(BASE_PATH, "ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated")

OUTPUT_MODEL_DIR = os.path.join(BASE_PATH, "meds-tab_check/models_new_all-1_f2")
TASKS_DIR = os.path.join(BASE_PATH, "meds-tab_check/tasks_new_all-1_f2")

N_PARALLEL_WORKERS = 4
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

# Get all phenotype task subfolders
# phenotype_tasks = list(sorted([
#     name for name in os.listdir(PHENOTYPE_PATH)
#     if os.path.isdir(os.path.join(PHENOTYPE_PATH, name))
# ]))

phenotype_tasks = ['long_los/labels']

# phenotype_tasks = list(sorted(glob.glob(os.path.join(PHENOTYPE_PATH, "*.parquet"))))
# print(phenotype_tasks)
# exit()

for i, task in (pbar := tqdm(enumerate(phenotype_tasks))):
    if "logs" in task:
        continue
    # if "long_los" not in task:
        # continue
    # task = task.split('/')[0]  # Get the last part of the path
    task_path = os.path.join(PHENOTYPE_PATH, task)
    task = task.split('/')[0]
    pbar.set_description(task)
    task_path = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated/task_labels/long_los_sharded"


    # Step 1: Run reshard.py for each split
    # for split in SPLITS:
    #     print("Resharding", split)

    #     if task_path.endswith(".parquet"):
    #         task_path = os.path.dirname(task_path)
    #         local_split = task.replace(".parquet", "")
    #     else:
    #         local_split = split

    #     cohort_input = os.path.join(task_path, f"{local_split}.parquet")
    #     cohort_output = os.path.join(TASKS_DIR, task)
    #     if not os.path.exists(cohort_input):
    #         print(f"Warning: {cohort_input} not found. Skipping.")
    #         continue

    #     cmd_reshard = [
    #         "python",
    #         "/data/mchome/ffp2106/ehr_foundation_model_benchmark/src/ehr_foundation_model_benchmark/tutorials/meds-tab/reshard.py",
    #         # "/data/mchome/ffp2106/femr/src/femr/omop_meds_tutorial/reshard.py",
    #         "--cohort_input", cohort_input,
    #         "--meds_data", os.path.join(REDSHARD_DIR, "data"),
    #         "--cohort_output", cohort_output,
    #         "--split", split
    #     ]

    #     start_time = time.time()  # Start timing
    #     try:
    #         print("Running:", " ".join(cmd_reshard))
    #         # subprocess.run(cmd_reshard, check=True)
    #     except Exception as e:
    #         log_error(task, f"reshard.py ({split})", e)
    #     finally:
    #         duration = time.time() - start_time  # Calculate duration
    #         log_training_time(task, f"reshard.py ({split})", duration)

    # Step 2: Run meds-tab-describe
    describe_cmd = [
        "meds-tab-describe",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final')}"
    ]
    start_time = time.time()  # Start timing
    try:
        print("Running:", " ".join(describe_cmd))
        subprocess.run(describe_cmd, check=True)
    except Exception as e:
        log_error(task, "meds-tab-describe", e)
    finally:
        duration = time.time() - start_time  # Calculate duration
        log_training_time(task, "meds-tab-describe", duration)

    # Step 3: Run meds-tab-tabularize-time-series
    tabularize_cmd = [
        "meds-tab-tabularize-time-series",
        "--multirun",
        f"worker=range(0,{N_PARALLEL_WORKERS})",
        "hydra/launcher=joblib",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final')}",
        "do_overwrite=False",
        # f"input_label_dir={task_path}",
        "tabularization.min_code_inclusion_count=1",
        "tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]",
        "tabularization.window_sizes=[1d,7d,30d,60d,365d,full]"
    ]
    start_time = time.time()  # Start timing
    try:
        print("Running:", " ".join(tabularize_cmd))
        subprocess.run(tabularize_cmd, check=True)
    except Exception as e:
        log_error(task, "meds-tab-tabularize-time-series", e)
    finally:
        duration = time.time() - start_time  # Calculate duration
        log_training_time(task, "meds-tab-tabularize-time-series", duration)

    label_cmd = [
        "meds-tab-cache-task",
        "--multirun",
        f"worker=range(0,{N_PARALLEL_WORKERS})",
        "hydra/launcher=joblib",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        f"input_label_dir={task_path}",
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final')}",
        f"task_name={task + '_task'}",
        
        "do_overwrite=False",
        "tabularization.min_code_inclusion_count=1",
        "tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]",
        "tabularization.window_sizes=[1d,7d,30d,60d,365d,full]"
    ]
    start_time = time.time()  # Start timing
    try:
        print("Running:", " ".join(label_cmd))
        subprocess.run(label_cmd, check=True)
    except Exception as e:
        log_error(task, "meds-tab-cache-task", e)
    finally:
        duration = time.time() - start_time  # Calculate duration
        log_training_time(task, "meds-tab-cache-task", duration)

    # Step 4: Run meds-tab-xgboost
    # for ratio in [0.001, 0.01, 0.1]:
    xgboost_cmd = [
        "meds-tab-xgboost",
        "--multirun",
        f"worker=range(0,{N_PARALLEL_WORKERS})",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final')}",
        f"output_model_dir={os.path.join(OUTPUT_MODEL_DIR, task)}",
        f"task_name={task + '_task'}",
        "do_overwrite=False",
        "hydra.sweeper.n_trials=10",
        f"hydra.sweeper.n_jobs={N_PARALLEL_WORKERS}",
        # f"input_tabularized_cache_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final', 'tabularize')}",
        "tabularization.min_code_inclusion_count=1",
        # f"input_label_cache_dir={task_path}",
        "tabularization.aggs=[code/count]",
        "tabularization.window_sizes=[1d,7d,30d,60d,365d,full]",
        # f"+model_launcher.model.stratify={ratio}"
    ]
    start_time = time.time()  # Start timing
    try:
        print("Running:", " ".join(xgboost_cmd))
        subprocess.run(xgboost_cmd, check=True)
    except Exception as e:
        log_error(task, f"meds-tab-xgboost", e)
    finally:
        duration = time.time() - start_time  # Calculate duration
        log_training_time(task, f"meds-tab-xgboost", duration)

print("Done")
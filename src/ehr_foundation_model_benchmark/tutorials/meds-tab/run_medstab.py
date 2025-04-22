import os
import subprocess
import traceback
from datetime import datetime
from tqdm import tqdm
import glob

# Constants
BASE_PATH = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped"
# PHENOTYPE_PATH = os.path.join(BASE_PATH, "task_labels/in_house_phenotypes/phenotype_cohorts_min_obs_2_years")
PHENOTYPE_PATH = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/motor/labels"
REDSHARD_DIR = os.path.join(BASE_PATH, "post_transform")
OUTPUT_MODEL_DIR = os.path.join(BASE_PATH, "models/meds_tab/output")
TASKS_DIR = os.path.join(BASE_PATH, "models/meds_tab/labels")
N_PARALLEL_WORKERS = 4
SPLITS = ["train", "tuning", "held_out"]
LOG_FILE = "processing_errors.log"

def log_error(task, step, exception):
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{datetime.now()}] ERROR in Task: {task} | Step: {step}\n")
        f.write(f"{traceback.format_exc()}\n")

# Get all phenotype task subfolders
# phenotype_tasks = list(sorted([
#     name for name in os.listdir(PHENOTYPE_PATH)
#     if os.path.isdir(os.path.join(PHENOTYPE_PATH, name))
# ]))

phenotype_tasks = list(sorted(glob.glob(os.path.join(PHENOTYPE_PATH, "*.parquet"))))
print(phenotype_tasks)
# exit()

for task in (pbar := tqdm(phenotype_tasks)):
    if "long_los" not in task:
        continue
    task = os.path.basename(task)
    task_path = os.path.join(PHENOTYPE_PATH, task)
    pbar.set_description(task)

    # Step 1: Run reshard.py for each split
    for split in SPLITS:
        print("Resharding", split)

        if task_path.endswith(".parquet"):
            task_path = os.path.dirname(task_path)
            local_split = task.replace(".parquet", "")

        cohort_input = os.path.join(task_path, f"{local_split}.parquet")
        cohort_output = os.path.join(TASKS_DIR, task)
        if not os.path.exists(cohort_input):
            print(f"Warning: {cohort_input} not found. Skipping.")
            continue

        cmd_reshard = [
            "python", "/data/mchome/ffp2106/femr/src/femr/omop_meds_tutorial/reshard.py",
            "--cohort_input", cohort_input,
            "--meds_data", os.path.join(REDSHARD_DIR, "data"),
            "--cohort_output", cohort_output,
            "--split", split
        ]

        try:
            print("Running:", " ".join(cmd_reshard))
            subprocess.run(cmd_reshard, check=True)
        except Exception as e:
            log_error(task, f"reshard.py ({split})", e)

    # Step 2: Run meds-tab-describe
    describe_cmd = [
        "meds-tab-describe",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final')}"
    ]
    try:
        print("Running:", " ".join(describe_cmd))
        subprocess.run(describe_cmd, check=True)
    except Exception as e:
        log_error(task, "meds-tab-describe", e)

    # Step 3: Run meds-tab-tabularize-time-series
    tabularize_cmd = [
        "meds-tab-tabularize-time-series",
        "--multirun",
        f"worker=range(0,{N_PARALLEL_WORKERS})",
        "hydra/launcher=joblib",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final')}",
        "do_overwrite=False",
        f"input_label_dir={os.path.join(TASKS_DIR, task)}",
        "tabularization.min_code_inclusion_count=10",
        "tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]",
        "tabularization.window_sizes=[1d,7d,30d,60d,365d,full]"
    ]
    try:
        print("Running:", " ".join(tabularize_cmd))
        subprocess.run(tabularize_cmd, check=True)
    except Exception as e:
        log_error(task, "meds-tab-tabularize-time-series", e)

    # Step 4: Run meds-tab-xgboost
    xgboost_cmd = [
        "meds-tab-xgboost",
        "--multirun",
        f"worker=range(0,{N_PARALLEL_WORKERS})",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final')}",
        f"output_model_dir={os.path.join(OUTPUT_MODEL_DIR, task)}",
        f"task_name={task}",
        "do_overwrite=False",
        "hydra.sweeper.n_trials=10",
        f"hydra.sweeper.n_jobs={N_PARALLEL_WORKERS}",
        f"input_tabularized_cache_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final', 'tabularize')}",
        "tabularization.min_code_inclusion_count=10",
        f"input_label_cache_dir={os.path.join(TASKS_DIR, task)}",
        "tabularization.aggs=[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]",
        "tabularization.window_sizes=[1d,7d,30d,60d,365d,full]"
    ]
    try:
        print("Running:", " ".join(xgboost_cmd))
        subprocess.run(xgboost_cmd, check=True)
    except Exception as e:
        log_error(task, "meds-tab-xgboost", e)
    
print("Done")
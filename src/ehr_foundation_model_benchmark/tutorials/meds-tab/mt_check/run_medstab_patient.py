import os
import subprocess
import traceback
import time  # Add this import
from datetime import datetime
from tqdm import tqdm
import glob

if __name__ == "__main__":
    BASE_PATH = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/"
    PHENOTYPE_PATH = os.path.join(BASE_PATH, "ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated_output/long_los/labels")
    REDSHARD_DIR = os.path.join(BASE_PATH, "ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated")
    # arg parse for featurization strategy
    import argparse

    parser = argparse.ArgumentParser(description="Run meds-tab patient processing.")
    parser.add_argument(
        "--featurization_strategy",
        type=str,
        default="general",
        help="Featurization strategy to use (default: general)",
    )
    # feature aggregation level
    parser.add_argument(
        "--feature_aggregation_level",
        type=str,
        default="tiny",
        help="Feature aggregation level to use (default: tiny)",
    )
    # version
    parser.add_argument(
        "--version",
        type=str,
        default="0.1",
        help="Version of the processing script (default: 0.1)",
    )
    # number of trials
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter tuning (default: 100)",
    )
    
    # label folder
    parser.add_argument(
        "--label_folder",
        type=str,
        default=PHENOTYPE_PATH,
        help=f"Folder containing the labels (default: {PHENOTYPE_PATH})",
    )
    args = parser.parse_args()

    if args.feature_aggregation_level == 'tiny':
        windows = "[full]"
        aggs = '[code/count]'
    else:
        windows = "[1d,7d,30d,60d,365d,full]"
        aggs = '[code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]'

    
    # OUTPUT_MODEL_DIR = os.path.join(BASE_PATH, f"meds-tab_check/models_mic0_{args.featurization_strategy}_{args.feature_aggregation_level}_{args.version}_{args.n_trials}")
    # TASKS_DIR = os.path.join(BASE_PATH, f"meds-tab_check/tasks_mic0_{args.featurization_strategy}_{args.feature_aggregation_level}_{args.version}_{args.n_trials}")

    # OUTPUT_MODEL_DIR = os.path.join(BASE_PATH, f"meds-tab_check/models_micn_{args.featurization_strategy}_{args.feature_aggregation_level}_{args.version}_{args.n_trials}")
    # TASKS_DIR = os.path.join(BASE_PATH, f"meds-tab_check/tasks_micn_{args.featurization_strategy}_{args.feature_aggregation_level}_{args.version}_{args.n_trials}")

    

    OUTPUT_MODEL_DIR = os.path.join(BASE_PATH, f"meds-tab_check/models_mic1_{args.featurization_strategy}_{args.feature_aggregation_level}_{args.version}_{args.n_trials}")
    TASKS_DIR = os.path.join(BASE_PATH, f"meds-tab_check/tasks_mic1_{args.featurization_strategy}_{args.feature_aggregation_level}_{args.version}_{args.n_trials}")

    N_PARALLEL_WORKERS = 4

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

    task = "long_los"
    # task_path = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated/task_labels/long_los_sharded"

    # Step 2: Run meds-tab-describe
    describe_cmd = [
        "meds-tab-describe",
        f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
        # f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output')}"
        f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output-test')}"

    ]
    start_time = time.time()  # Start timing
    try:
        print("Running:", " ".join(describe_cmd))
        subprocess.run(describe_cmd, check=True)
    except Exception as e:
        log_error(task, "meds-tab-describe", e)
        raise e
    finally:
        duration = time.time() - start_time  # Calculate duration
        log_training_time(task, "meds-tab-describe", duration)

    if args.featurization_strategy == "general":
        # Step 3: Run meds-tab-tabularize-time-series
        tabularize_cmd = [
            "meds-tab-tabularize-time-series",
            "--multirun",
            f"worker=range(0,{N_PARALLEL_WORKERS})",
            "hydra/launcher=joblib",
            f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
            f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output-test')}",

            # f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output')}",
            "do_overwrite=False",
            # f"input_label_dir={task_path}",
            "tabularization.min_code_inclusion_count=1",
            f"tabularization.aggs={aggs}",
            f"tabularization.window_sizes={windows}",
        ]
        start_time = time.time()  # Start timing
        try:
            print("Running:", " ".join(tabularize_cmd))
            subprocess.run(tabularize_cmd, check=True)
        except Exception as e:
            log_error(task, "meds-tab-tabularize-time-series", e)
            raise e
        finally:
            duration = time.time() - start_time  # Calculate duration
            log_training_time(task, "meds-tab-tabularize-time-series", duration)

        label_cmd = [
            "meds-tab-cache-task",
            "--multirun",
            f"worker=range(0,{N_PARALLEL_WORKERS})",
            "hydra/launcher=joblib",
            f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
            f"input_label_dir={args.label_folder}",
            f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output-test')}",

            # f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output')}",
            f"task_name={task}",
            
            "do_overwrite=False",
            "tabularization.min_code_inclusion_count=1",
            f"tabularization.aggs={aggs}",
            f"tabularization.window_sizes={windows}",
        ]
        start_time = time.time()  # Start timing
        try:
            print("Running:", " ".join(label_cmd))
            subprocess.run(label_cmd, check=True)
        except Exception as e:
            log_error(task, "meds-tab-cache-task", e)
            raise e
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
            # f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output')}",
            f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output-test')}",
            f"output_model_dir={os.path.join(OUTPUT_MODEL_DIR, task)}",
            f"task_name={task}",
            "do_overwrite=False",
            f"hydra.sweeper.n_trials={args.n_trials}",
            f"hydra.sweeper.n_jobs={N_PARALLEL_WORKERS}",
            # f"input_tabularized_cache_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final', 'tabularize')}",
            "tabularization.min_code_inclusion_count=1",
            # f"input_label_cache_dir={task_path}",
            f"tabularization.aggs={aggs}",
            f"tabularization.window_sizes={windows}",
            # f"+model_launcher.model.stratify={ratio}"
        ]
        start_time = time.time()  # Start timing
        try:
            print("Running:", " ".join(xgboost_cmd))
            subprocess.run(xgboost_cmd, check=True)
        except Exception as e:
            log_error(task, f"meds-tab-xgboost", e)
            raise e
        finally:
            duration = time.time() - start_time  # Calculate duration
            log_training_time(task, f"meds-tab-xgboost", duration)
    else:
        # Step 3: Run meds-tab-tabularize-time-series
        tabularize_cmd = [
            "meds-tab-tabularize-time-series",
            "--multirun",
            f"worker=range(0,{N_PARALLEL_WORKERS})",
            "hydra/launcher=joblib",
            f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
            f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output-test')}",
            # f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output')}",
            "do_overwrite=False",
            f"input_label_dir={args.label_folder}",
            "tabularization.min_code_inclusion_count=1",
            f"tabularization.aggs={aggs}",
            f"tabularization.window_sizes={windows}",
        ]
        start_time = time.time()  # Start timing
        try:
            print("Running:", " ".join(tabularize_cmd))
            subprocess.run(tabularize_cmd, check=True)
        except Exception as e:
            log_error(task, "meds-tab-tabularize-time-series", e)
            raise e
        finally:
            duration = time.time() - start_time  # Calculate duration
            log_training_time(task, "meds-tab-tabularize-time-series", duration)

        # Step 4: Run meds-tab-xgboost
        # for ratio in [0.001, 0.01, 0.1]:
        xgboost_cmd = [
            "meds-tab-xgboost",
            "--multirun",
            f"worker=range(0,{N_PARALLEL_WORKERS})",
            f"input_dir={os.path.join(REDSHARD_DIR, 'data')}",
            f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output-test')}",

            # f"output_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output')}",
            f"output_model_dir={os.path.join(OUTPUT_MODEL_DIR, task)}",
            f"task_name={task}",
            "do_overwrite=False",
            f"hydra.sweeper.n_trials={args.n_trials}",
            f"hydra.sweeper.n_jobs={N_PARALLEL_WORKERS}",
            # f"input_tabularized_cache_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_final', 'tabularize')}",
            "tabularization.min_code_inclusion_count=1",
            # f"input_label_cache_dir={task_path}"
            f"input_tabularized_cache_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output-test', 'tabularize')}",
            
            # f"input_tabularized_cache_dir={os.path.join(OUTPUT_MODEL_DIR, task + '_output', 'tabularize')}",
            f"input_label_cache_dir={args.label_folder}",

            f"tabularization.aggs={aggs}",
            f"tabularization.window_sizes={windows}",
            # f"+model_launcher.model.stratify={ratio}"
        ]
        start_time = time.time()  # Start timing
        try:
            print("Running:", " ".join(xgboost_cmd))
            subprocess.run(xgboost_cmd, check=True)
        except Exception as e:
            log_error(task, f"meds-tab-xgboost", e)
            raise e
        finally:
            duration = time.time() - start_time  # Calculate duration
            log_training_time(task, f"meds-tab-xgboost", duration)

    print("Done")

# command line
# python run_medstab_patient.py --featurization_strategy general --feature_aggregation_level tiny --version 0.1 --n_trials 10 --label_folder /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated/task_labels/long_los_sharded 
# 

# python run_medstab_patient.py --featurization_strategy task --feature_aggregation_level tiny --version 0.1 --n_trials 100 --label_folder /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated/task_labels/long_los_sharded 


# python run_medstab_patient.py --featurization_strategy general --feature_aggregation_level tiny --version 0.0.9 --n_trials 10 --label_folder /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated/task_labels/long_los_sharded 
# python run_medstab_patient.py --featurization_strategy general --feature_aggregation_level tiny --version 0.0.5 --n_trials 10 --label_folder /data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated/task_labels/long_los_sharded 

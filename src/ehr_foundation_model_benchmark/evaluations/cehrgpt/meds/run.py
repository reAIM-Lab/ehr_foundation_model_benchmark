import os
import subprocess
import sys
import yaml

# ===============================
# LOAD YAML CONFIG
# ===============================

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ===============================
# MAIN PIPELINE
# ===============================

def main():
    config_name = sys.argv[1]
    cfg = load_config(f"{os.path.dirname(__file__)}/{config_name}.yaml")

    # Extract variables
    CEHR_GPT_MODEL_DIR = cfg["CEHR_GPT_MODEL_DIR"]
    CEHR_GPT_DATA_DIR = cfg["CEHR_GPT_DATA_DIR"]
    CEHR_GPT_PREPARED_DATA_DIR = cfg["CEHR_GPT_PREPARED_DATA_DIR"]
    TOKENIZED_PATH = cfg["tokenized_full_dataset_path"]
    COHORTS_ROOT = cfg["cohorts_root"]
    OUTPUT_BASE = cfg["output_base"]
    RESULTS_BASE = cfg["results_base"]
    LINEAR_PROB_SCRIPT = os.path.expanduser(cfg["linear_prob_script"])
    observation_window = cfg['observation_window']
    observation_window_ignore = cfg['observation_window_ignore']
    exclude_tables = cfg['exclude_tables']

    # Print config summary
    print("=== Loaded Configuration ===")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("============================\n")

    # Discover cohort folders
    cohort_folders = [
        os.path.join(COHORTS_ROOT, d)
        for d in os.listdir(COHORTS_ROOT)
        if os.path.isdir(os.path.join(COHORTS_ROOT, d))
    ]

    print(f"Found {len(cohort_folders)} cohorts:")
    for f in cohort_folders:
        print("  -", f)

    # Process each cohort
    for cohort_folder in cohort_folders:
        try:
            cohort_name = os.path.basename(cohort_folder)
            print(f"\n=== Processing cohort: {cohort_name} ===")

            output_dir = os.path.join(OUTPUT_BASE, cohort_name)
            results_dir = RESULTS_BASE
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)

            # -------------------------
            # Step 1: Compute CEHR-GPT features
            # -------------------------
            cmd_compute = [
                "python", "-u", "-m", "cehrgpt.tools.linear_prob.compute_cehrgpt_features",
                "--is_data_in_meds",
                "--dataset_prepared_path", CEHR_GPT_PREPARED_DATA_DIR,
                "--tokenized_full_dataset_path", TOKENIZED_PATH,
                "--model_name_or_path", CEHR_GPT_MODEL_DIR,
                "--tokenizer_name_or_path", CEHR_GPT_MODEL_DIR,
                "--data_folder", CEHR_GPT_DATA_DIR,
                "--cohort_folder", cohort_folder,
                "--inpatient_att_function_type", "day",
                "--att_function_type", "day",
                "--include_inpatient_hour_token", "true",
                "--include_auxiliary_token", "true",
                "--include_demographic_prompt", "true",
                "--disconnect_problem_list_events", "true",
                "--meds_to_cehrbert_conversion_type", "MedsToCehrbertOMOP",
                "--output_dir", output_dir,
                "--preprocessing_num_workers", "8",
                "--per_device_eval_batch_size", "8",
                "--max_tokens_per_batch", "16384",
                "--sample_packing"
            ]
            if exclude_tables:
                cmd_compute += ["--meds_exclude_tables", "measurement observation device_exposure"]
            if observation_window is not None:
                if observation_window_ignore is None or cohort_name not in observation_window_ignore:
                    cmd_compute = cmd_compute + ["--observation_window", str(observation_window)]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            print("🚀 Running CEHR-GPT feature computation...")
            subprocess.run(cmd_compute, check=True, env=env)

            # -------------------------
            # Step 2: Run linear probing
            # -------------------------
            cmd_linear = [
                "sh", LINEAR_PROB_SCRIPT,
                "--base_dir", output_dir,
                "--output_dir", results_dir,
                "--meds_dir", CEHR_GPT_DATA_DIR,
                "--model_name", "cehrgpt"
            ]

            print("⚙️ Running linear probing few-shot evaluation...")
            subprocess.run(cmd_linear, check=True)

            print(f"✅ Completed cohort: {cohort_name}\n")
        except Exception as e:
            print("Failed")
            import traceback
            print(traceback.format_exc())


if __name__ == "__main__":
    main()

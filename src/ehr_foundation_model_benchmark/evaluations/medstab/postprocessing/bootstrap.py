"""
Code to export the xgboost results
"""
import os
import subprocess
import shutil

# Root path where all 'xxx' folders are located
root_dir = "XX4"

print(os.listdir(root_dir))

# Base command to call
base_script = "meds-evaluation/src/meds_evaluation/__main__.py"

# Output base directory for results
results_base = "meds_evaluation/results_final"

# Loop over all folders in root_dir
# for sampling in [0.001, 0.01, 0.1]:
for sampling in [1.0]:
    for model_name in os.listdir(root_dir):
        if "_final" in model_name:
            continue  # Skip folders with '_final'
        if not str(sampling) in model_name:
            continue
            
        try:
            model_path = os.path.join(root_dir, model_name)
            if not os.path.isdir(model_path):
                continue  # Just to be safe, skip non-directories

            # There should be exactly one subfolder (date-time folder)
            subfolders = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]

            # if len(subfolders) != 1:
            #     print(f"[Warning] Skipping {model_path}: expected exactly one subfolder, found {len(subfolders)}.")
            #     continue

            # if several subfolders, take the last one, ordered by name
            if len(subfolders) == 0:
                print(f"[Warning] No subfolders found in {model_path}. Skipping.")
                continue
            subfolders.sort()

            # datetime_folder = subfolders[0]
            datetime_folder = subfolders[-1]
            
            # Build full predictions_path
            predictions_path = os.path.join(model_path, datetime_folder, "best_trial", "held_out_predictions.parquet")


            if not os.path.exists(os.path.expanduser(predictions_path)):
                print(f"[Warning] Predictions file not found at {predictions_path}. Skipping.")
                continue


            task = model_name.replace("-1.0", "")
            output_to_copy = f'XX7/{task}'
            os.makedirs(output_to_copy, exist_ok=True)
            shutil.copy2(predictions_path, f'{output_to_copy}/medstab_100000.parquet')


            # Output folder for results (use model name)
            output_path = os.path.join(results_base, model_name)

            if os.path.exists(output_path):
                print(f"[Warning] Output path {output_path} already exists. Skipping.")
                continue

            # Build the full command
            command = f"python {base_script} predictions_path=\"{predictions_path}\" output_dir=\"{output_path}\""

            print(f"[Running] {command}")

            # Execute the command
            subprocess.run(command, shell=True, check=True)
        except Exception as e:
            print(e)

print("\nAll evaluations done!")

"""
Code to export the linear results
"""

import os
import subprocess
import shutil

root_dir = "XX7"

print(os.listdir(root_dir))

# Base command to call
base_script = "meds-evaluation/src/meds_evaluation/__main__.py"

# Output base directory for results
results_base = "src/meds_evaluation/results_final"

# Loop over all folders in root_dir
# for sampling in [0.001, 0.01, 0.1]:
for sampling in [1.0]:
    for model_name in os.listdir(root_dir):
        if "_final" in model_name:
            continue  # Skip folders with '_final'
            
        try:
            model_path = os.path.join(root_dir, model_name)
            if not os.path.isdir(model_path):
                continue  # Just to be safe, skip non-directories

            # Build full predictions_path
            predictions_path = os.path.join(model_path, "medstab-lr_100000.parquet")

            if not os.path.exists(os.path.expanduser(predictions_path)):
                print(f"[Warning] Predictions file not found at {predictions_path}. Skipping.")
                continue

            # Output folder for results (use model name)
            output_path = os.path.join(results_base, model_name + "-lr")

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

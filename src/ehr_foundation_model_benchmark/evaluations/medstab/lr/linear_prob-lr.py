
"""
Code to run logistic regressions for 100, 1000, 10k and 100k (all) samples
"""
import subprocess

# tasks = ["long_los", "death", "readmission"]
tasks = ['Ischemic_Stroke'] # edit your tasks

base_command = [
    "python", "lr/linear_prob.py",
    "--features_label_input_dir", "XX4/tabularize_export",
    "--meds_dir", "XX1",
    "--output_dir", "XX7",
    "--model_name", "medstab-lr",
    "--task_name", "{task}"
]

for task in tasks:
    print(f"Running task: {task}")
    command = [arg.format(task=task) for arg in base_command]
    subprocess.run(" ".join(command), shell=True, check=True)
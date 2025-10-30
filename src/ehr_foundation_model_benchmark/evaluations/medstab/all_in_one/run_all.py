import yaml
import sys
from pprint import pprint
import subprocess
import os

def load_yaml(file_path):
    """Load and return YAML content from a file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Ensure the user provides a file path
    if len(sys.argv) < 2:
        print("Usage: python load_yaml.py <path_to_yaml_file>")
        sys.exit(1)

    yaml_path = sys.argv[1]
    
    try:
        data = load_yaml(yaml_path)

        for key in data:
            if key.endswith("_dir"):
                data[key] = os.path.join(data['base_path'], data[key])

        print("✅ YAML content loaded successfully:\n")
        pprint(data)

        def get_tasks():
            all_tasks = list(sorted([
                name for name in os.listdir(data['phenotypes_dir'])
                if os.path.isdir(os.path.join(data['phenotypes_dir'], name))
            ]))
            if data['include'] is not None:
                all_tasks = [task for task in all_tasks if task in data['include']]
            if data['exclude'] is not None:
                all_tasks = [task for task in all_tasks if task not in data['exclude']]
            return all_tasks

        tasks = get_tasks()
        tasks_str = ','.join(tasks)
    except FileNotFoundError:
        print(f"❌ Error: File '{yaml_path}' not found.")
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error: {e}")

    cmd = [
        "python",
        "featurization/run_medstab_cmd.py",   # 👈 this is the featurization script (the third file)
        "--phenotypes-dir", data['phenotypes_dir'],
        "--data-dir", data['data_dir'],
        "--output-features-dir", data['output_features_dir'],
        "--output-reshard-dir", data['output_reshard_dir'],
        "--windows", data['windows'],
        "--tasks", tasks_str
    ]

    print()
    print("🚀 Running featurization command:\n", " ".join(cmd))
    print()

    # ---- Execute or dry-run ----
    if not data['dry_run']:
        subprocess.run(cmd, check=True)
        pass
    else:
        print("💡 Dry run: command not executed.")


    cmd = ["python", 'xgb/auto_sweep.py',
        "--phenotypes-dir", data["phenotypes_dir"],
        "--data-dir", data["data_dir"],
        "--output-features-dir", data.get("output_features_dir", ""),
        "--output_few_shots_dir", data.get("output_few_shots_dir", ""),
        "--output-reshard-dir", data.get("output_reshard_dir", ""),
        "--output-model-dir", data.get("output_model_dir", ""),
        "--output-results-dir", data.get("output_results_dir", ""),
        "--windows", data["windows"],
        "--tasks", tasks_str
    ]

    # Remove any empty args (in case some dirs aren't defined)
    cmd = [arg for arg in cmd if arg != ""]

    print("🚀 Running XGB training command:\n", " ".join(cmd))
    print()

    # ---- Execute or dry-run ----
    if not data['dry_run']:
        subprocess.run(cmd, check=True)
        pass
    else:
        print("💡 Dry run: command not executed.")


    print("🚀 Running LR training command:\n", " ".join(cmd))
    print()
    for task in tasks:
        command = [
            "python", "lr/linear_prob.py",
            "--features_label_input_dir", os.path.join(data['output_features_dir'], f"{task}_final", 'tabularize_export'),
            "--meds_dir", data['data_dir'],
            "--output_dir", data.get("output_results_dir", ""),
            "--model_name", "medstab-lr",
            "--task_name", f"{task}"
        ]
        print(command)
        if not data['dry_run']:
            subprocess.run(" ".join(command), shell=True, check=True)
        else:
            print("💡 Dry run: command not executed.")

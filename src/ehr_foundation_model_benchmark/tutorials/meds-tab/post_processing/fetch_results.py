import os
import json
import pandas as pd

results_base = "/data/mchome/ffp2106//meds-evaluation/src/meds_evaluation/results_final"


results = []

for model_name in os.listdir(os.path.expanduser(results_base)):
    model_dir = os.path.join(os.path.expanduser(results_base), model_name)
    if not os.path.isdir(model_dir):
        continue

    if "1.0" in model_name or "10.0" in model_name:
        print(f"[Skipping] {model_name} due to sampling ratio.")
        continue

    results_file = os.path.join(model_dir, "results_boot.json")
    if not os.path.exists(results_file):
        print(f"[Warning] results_boot.json not found for {model_name}.")
        continue

    with open(results_file, 'r') as f:
        data = json.load(f)

    mean_roc_auc = data.get("mean_roc_auc_score", None)
    std_roc_auc = data.get("std_roc_auc_score", None)

    results.append({
        "Model": model_name,
        "Mean ROC AUC": mean_roc_auc,
        "Std ROC AUC": std_roc_auc
    })

# Create and print dataframe
results_df = pd.DataFrame(results)
# lower Model
results_df["Model"] = results_df["Model"].str.lower()
results_df = results_df.sort_values(by="Model")
print("\nSummary of ROC AUC scores:")
print(results_df)

print("\nSummary of ROC AUC scores:")
for res in results_df.to_dict(orient='records'):
    if res["Mean ROC AUC"] is not None and res["Std ROC AUC"] is not None:
        print(f"{res['Model']} {res['Mean ROC AUC']:.3f} ± {res['Std ROC AUC']:.3f}")

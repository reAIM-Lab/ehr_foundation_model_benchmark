import os
import json
import pandas as pd

# Define input folder and output file
input_folder = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/tabular"  # Update with your actual folder path
output_file = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/femr/tabular/compiled_results.csv"

# Initialize a list to store extracted data
data_list = []

# Iterate through files in the folder
for filename in os.listdir(input_folder):
    if ("_results_") in filename and filename.endswith(".json"):  # Ensure correct files are read
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                
                # Extract key name from filename (first part before '_')
                key_name = filename.split("_")[1]  # Extract the part after "lightgbm_"

                # Append extracted data to list with dynamic key
                data_list.append({
                    "filename": filename,
                    "label_name": data.get("label_name", "N/A"),
                    key_name: data.get("final_lightgbm_auroc", "N/A")
                })
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {filename}: {e}")

# Convert list to DataFrame
df = pd.DataFrame(data_list)

# Save DataFrame to CSV
df.to_csv(output_file, index=False)

print(f"Compiled results saved to {output_file}")
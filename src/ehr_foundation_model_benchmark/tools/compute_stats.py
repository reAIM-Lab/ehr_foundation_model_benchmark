from ehr_foundation_model_benchmark.tools.path import files
import pickle

for i, file in enumerate(files):
    # file = file.replace(".snappy.parquet", "-harmonized-harmonized-3.snappy.parquet")
    file = file.replace(".snappy.parquet", "-harmonized-v2.snappy.parquet")

with open('stats.pkl', 'rb') as f:
    val_df, val_a, val_b, val_c, val_d, val_e = pickle.load(f)

pct = lambda x: (sum(x), sum(x) / sum(val_df) * 100)

print("Non harmonized with value as number", pct(val_a))
print("Non harmonized with value as concept", pct(val_b))
print("Non harmonized with value as number or as concept", pct(val_c))
print("Non harmonized with value as number or as concept AND with NaN unit", pct(val_d))
print("Non harmonized with value as number or as concept having more than one unit", pct(val_e))


# compute incomplete measurement_concept_id

# read list of csv, concatenate then and order by two columns, the first descending, the second ascending

import pandas as pd
import glob

# Specify the pattern to match your CSV files
csv_files = glob.glob("agg-*.csv")  # Update "path_to_directory" with the folder containing your CSV files

# Read and concatenate all CSV files
dfs = [pd.read_csv(file, index_col=0) for file in csv_files]

# Check if there are any CSV files to process
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # Sort by two columns: 'column1' descending and 'column2' ascending
    # Replace 'column1' and 'column2' with your actual column names
    sorted_df = combined_df.sort_values(by=['total_measurements', 'measurement_concept_id'], ascending=[False, True])

    # Display the sorted DataFrame
    print(sorted_df)
    sorted_df = sorted_df.reset_index(drop=True)
    sorted_df.to_csv('agg_all.csv')
else:
    print("No CSV files found. Please check the directory or file paths.")
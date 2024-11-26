
import pandas as pd
from path import files

total_rows_acc = 0
missing_values_acc = 0
for file in files:
    file = file.replace(".snappy.parquet", "-harmonized.snappy.parquet")
    print()
    print("File", file)
    
    df = pd.read_parquet(file)
    # print(df.columns)
    # print(df[['measurement_concept_id', 'unit_concept_name', 'value_as_number', 'harmonized_value_as_number',
    #     'harmonized_unit_concept_id']])

    # Count the number of rows in the DataFrame
    total_rows = len(df)
    total_rows_acc += total_rows

    # Count the number of missing values in the specified column
    missing_values = df['harmonized_value_as_number'].isnull().sum()
    missing_values_acc += missing_values

    # Calculate the percentage of missing values
    percentage_missing = (missing_values / total_rows) * 100

    print(f"Total rows: {total_rows}")
    print(f"Missing values: {missing_values}")
    print(f"Percentage of missing values: {percentage_missing:.2f}%")


percentage_missing_acc = (missing_values_acc / total_rows_acc) * 100

print(f"Total rows: {total_rows_acc}")
print(f"Missing values: {missing_values_acc}")
print(f"Percentage of missing values: {percentage_missing_acc:.2f}%")
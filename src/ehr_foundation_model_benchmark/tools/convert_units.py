
import pandas as pd
from ehr_foundation_model_benchmark.src.ehr_foundation_model_benchmark.tools.mappings import mapping_functions, get_unit_id, get_measurements, get_one_unit_lab
import numpy as np
from tqdm import tqdm

from path import file, file_out

print("Loading")

data = pd.read_parquet(file)
print(data.columns)

data['harmonized_value_as_number'] = None
data['harmonized_unit_concept_id'] = None

for measurement_id in tqdm(get_one_unit_lab(), desc="Single unit labs"):
    cdt = data['measurement_concept_id'] == measurement_id

    data.loc[cdt, 'harmonized_value_as_number'] = data.loc[cdt, 'value_as_number']
    data.loc[cdt, 'harmonized_unit_concept_id'] = data.loc[cdt, 'unit_concept_id']

for mapping_key, mapping_fun in tqdm(mapping_functions.items(), desc="Multi unit labs"):
    cdt = (data['unit_concept_id'] == get_unit_id(mapping_key[0])) & data['measurement_concept_id'].isin(get_measurements(mapping_key))
    print("Converting", mapping_key, np.count_nonzero(cdt))
    
    data.loc[cdt, 'harmonized_value_as_number'] = data.loc[cdt, 'value_as_number'].apply(mapping_fun)
    data.loc[cdt, 'harmonized_unit_concept_id'] = get_unit_id(mapping_key[1])

print(data)
print(data[data['unit_concept_id'] == get_unit_id(mapping_key[0])])
print("Harmonized units percentage", 100 - data['harmonized_value_as_number'].isnull().sum() * 100 / len(data)) # or use mean https://stackoverflow.com/questions/51070985/find-out-the-percentage-of-missing-values-in-each-column-in-the-given-dataset

# print('save')
# data.to_parquet(file_out)
# print('end')

# next time:
# - add unit equivalent
# - add most common determination
# - add unit removal
# - continue mapping
# - implement partial saving to avoid recomputing everything
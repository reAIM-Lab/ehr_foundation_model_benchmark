import json    
import pandas as pd

from ehr_foundation_model_benchmark.tools.mappings import (
    get_conversions,
    get_copy_majority_units,
    get_one_unit_lab,
    compute_most_common_units,
    # get_one_unit_and_missing_lab,
    convert_mappings_to_id,
    get_rare_units_labs,
    simplify_equivalent_units,
)
from ehr_foundation_model_benchmark.tools.path import files


    # print('Getting majority/minority and value type counts')
    # with open('most_common_units.json', 'r') as f:
    #     most_common_units = json.load(f)
for file in files:
    file = file.replace(".snappy.parquet", "-harmonized-v5.snappy.parquet")
    print()
    print("File", file)
    
    data = pd.read_parquet(file)

    most_common_units = compute_most_common_units()
        
    most_common_units_df = pd.Series(most_common_units).reset_index()
    most_common_units_df['index'] = most_common_units_df['index'].astype(int)
    most_common_units_df.set_index('index', inplace=True)
    most_common_units = most_common_units_df.to_dict()[0]
    
    #get majority unit
    print("Getting majority units")
    data['majority_unit_id'] = pd.DataFrame(data['measurement_concept_id']).replace(to_replace=most_common_units)['measurement_concept_id']
    
    
    print('Finished filling in majority units')
    # fill in unit match type: majority, minority, nan (acceptable, unacceptable)
    data['unit_match_type'] = 'to_fill'
    data['harmonized_unit_concept_id'].fillna(0, inplace=True)
    data.loc[(data['harmonized_unit_concept_id'] == data['majority_unit_id']) & (data['majority_unit_id'] == 0), 'unit_match_type'] = 'Mapped to majority unit (nan)'
    data.loc[(data['harmonized_unit_concept_id'] == data['majority_unit_id']) & (data['majority_unit_id'] != 0), 'unit_match_type'] = 'Mapped to majority unit (non-nan)'
    data.loc[(data['harmonized_unit_concept_id'] != data['majority_unit_id']) & (data['majority_unit_id'] != 0), 'unit_match_type'] = 'Mapped to minority unit'
    data.loc[(data['harmonized_unit_concept_id'] == 0) & (data['majority_unit_id'] != 0), 'unit_match_type'] = 'Unmapped nan'
    
    # fill in concept id type
    data['value_match_type'] = 'to_fill'
    data.loc[~(data['harmonized_value_as_number'].isna()), 'value_match_type'] = 'Numerical value'
    data.loc[(data['harmonized_value_as_number'].isna()) & ~((data['value_as_concept_id'].isna())), 'value_match_type'] = 'Concept value (no number)'
    data.loc[(data['harmonized_value_as_number'].isna()) & ((data['value_as_concept_id'].isna())), 'value_match_type'] = 'Nan concept and number'

    # added: is the original unit concept ID nan/0? 
    data['original_unit_type'] = 'Non-Nan'
    data.loc[((data['unit_concept_id'].isna()) | (data['unit_concept_id']==0)), 'original_unit_type'] = 'Nan'

    data['original_value_type'] = 'to_fill'
    data.loc[~(data['value_as_number'].isna()), 'original_value_type'] = 'Numerical value'
    data.loc[(data['value_as_number'].isna()) & ~((data['value_as_concept_id'].isna())), 'original_value_type'] = 'Concept value (no number)'
    data.loc[(data['value_as_number'].isna()) & ((data['value_as_concept_id'].isna())), 'original_value_type'] = 'Nan concept and number'

    
    counts = data.groupby(by=["unit_match_type", "value_match_type", "original_unit_type", 'original_value_type']).size().reset_index()
    
    counts.to_csv(file.replace(".snappy.parquet", "-v5-processed_unit_value_counts.csv"))
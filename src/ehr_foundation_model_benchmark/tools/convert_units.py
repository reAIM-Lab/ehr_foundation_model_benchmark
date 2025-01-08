import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from ehr_foundation_model_benchmark.tools.path import files
from ehr_foundation_model_benchmark.tools.mappings import (
    get_conversions,
    get_one_unit_lab,
    compute_most_common_units,
    get_one_unit_and_missing_lab,
    convert_mappings_to_id,
    get_rare_units_labs,
    simplify_equivalent_units,
)

### PREREQUISITES
### The file measurement measurement_unit_counts.csv should have been created before and be in the current working directory

# Demo mode does not process all the labs, only one for each pipeline step to check the pipeline runs
demo = False
one_file = True


def report_harmonized(data, name):
    print(
        f"Harmonized units percentage {name}",
        100 - data["harmonized_value_as_number"].isnull().sum() * 100 / len(data),
        "%",
    ) 


def process_file(file):
    print("Loading", file)
    data = pd.read_parquet(file)

    # new columns for the output
    data["harmonized_value_as_number"] = None
    data["harmonized_unit_concept_id"] = None

    # most_common_units = compute_most_common_units()

    # EQUIVALENT UNITS
    print("Simplifying units")
    data = simplify_equivalent_units(data)
    print("End simplifying unis")

    # UNDEFINED LABS
    print(len(data))
    if not demo:
        data = data.loc[data.measurement_concept_id != 0]  # check 130 millions

    # SINGLE UNIT LABS
    for measurement_id in tqdm(get_one_unit_lab(), desc="Single unit labs"):
        cdt = data["measurement_concept_id"] == measurement_id

        data.loc[cdt, "harmonized_value_as_number"] = data.loc[cdt, "value_as_number"]
        data.loc[cdt, "harmonized_unit_concept_id"] = data.loc[cdt, "unit_concept_id"]

        if demo:
            break

    report_harmonized(data, "after single unit labs")

    # ONE UNIT AND MISSING LABS
    # for measurement_id in tqdm(
    #     get_one_unit_and_missing_lab(), desc="Single unit + missing labs"
    # ):
    #     cdt = data["measurement_concept_id"] == measurement_id

    #     data.loc[cdt, "harmonized_value_as_number"] = data.loc[cdt, "value_as_number"]
    #     data.loc[cdt, "harmonized_unit_concept_id"] = most_common_units[measurement_id]

    #     if demo:
    #         break

    report_harmonized(data, "after single unit and missing labs")

    # MULTI-UNIT LABS - CONVERSION
    to_convert = get_conversions()
    mapping_functions_id = convert_mappings_to_id()
    for measurement_id, from_unit_id, to_unit_id in tqdm(
        to_convert, desc="Multi unit labs - conversion"
    ):
        # some conversions are only valid for specific measurement_concept
        # some conversions are valid no matter the measurement_concept
        cdt2 = (from_unit_id, to_unit_id, measurement_id) in mapping_functions_id
        if (from_unit_id, to_unit_id, None) in mapping_functions_id or cdt2:
            cdt = (data["unit_concept_id"] == from_unit_id) & (
                data["measurement_concept_id"] == measurement_id
            )

            # print(
            #     "Converting",
            #     measurement_id,
            #     from_unit_id,
            #     to_unit_id,
            #     np.count_nonzero(cdt),
            # )
            # count can be 0 because only one file here and not everything is loaded like for the measurement_unit_counts

            if cdt2:
                mapping_fun = mapping_functions_id[
                    (from_unit_id, to_unit_id, measurement_id)
                ]
            else:
                mapping_fun = mapping_functions_id[(from_unit_id, to_unit_id, None)]

            data.loc[cdt, "harmonized_value_as_number"] = data.loc[
                cdt, "value_as_number"
            ].apply(mapping_fun)
            data.loc[cdt, "harmonized_unit_concept_id"] = to_unit_id

        if demo:
            break

     # MULTI-UNIT LABS - COPY MAJORITY UNITS
     # it is modelled as a conversion from target_unit to target_unit
    already_done = []
    for measurement_id, from_unit_id, to_unit_id in tqdm(
        to_convert, desc="Multi unit labs - majority units"
    ):
        if not (measurement_id, to_unit_id) in already_done:
            cdt = (data["unit_concept_id"] == to_unit_id) & (
                data["measurement_concept_id"] == measurement_id
            )
            # print(
            #     "Converting",
            #     measurement_id,
            #     to_unit_id,
            #     to_unit_id,
            #     np.count_nonzero(cdt),
            # )
            mapping_fun = lambda x: x

            data.loc[cdt, "harmonized_value_as_number"] = data.loc[
                cdt, "value_as_number"
            ].apply(mapping_fun)
            data.loc[cdt, "harmonized_unit_concept_id"] = to_unit_id
            
        if demo:
            break

    report_harmonized(data, "after conversion")

    # RARE UNITS
    # for rare units, convert to nan (0) in the unit_concept_id
    rare_units = get_rare_units_labs()
    for measurement_id, unit_id in tqdm(rare_units, desc="Rare units"):
        cdt = (
            (data["unit_concept_id"] == unit_id)
            & (data["measurement_concept_id"] == measurement_id)
            & (data["harmonized_value_as_number"].isnull())
        )  # not already converted

        data.loc[cdt, "unit_concept_id"] = 0
        
        # column does not exist in data in fact, no need
        data.loc[cdt, "unit_concept_name"] = "No matching concept"

        if demo:
            break

    ###########################################
    # FURTHER PROCESSING NEEDED TO RESOLVE NANS
    ###########################################
    # TODO
    ###########################################

        print('Getting majority/minority and value type counts')
    f = open('most_common_units.json',)
    most_common_units = json.load(f)
    most_common_units_df = pd.Series(most_common_units).reset_index()
    most_common_units_df['index'] = most_common_units_df['index'].astype(int)
    most_common_units_df.set_index('index', inplace=True)
    most_common_units = most_common_units_df.to_dict()[0]
    
    #get majority unit
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
    
    counts = data.groupby(by=["unit_match_type", "value_match_type", "original_unit_type"]).size().reset_index()
    
    counts.to_csv(file.replace(".snappy.parquet", "-processed_unit_value_counts.csv"))


    report_harmonized(data, "at the end of the pipeline")

    if not demo:
        print("save")
        data.to_parquet(file.replace(".snappy.parquet", "-harmonized-v3.snappy.parquet"))
        # data.to_parquet(file.replace(".snappy.parquet", "-harmonized-v2-one-and-missing-2.snappy.parquet"))
        print("end")

    
if __name__ == "__main__":
    max_processes = 2 # 16 makes the server crash

    if demo:
        files = [files[0], files[1]]

    if one_file:
        files = [files[0]]

    with mp.get_context("spawn").Pool(processes=max_processes) as pool:
        results = pool.map(process_file, files)

    for result in results:
        print(f"File processed with result: {result}")

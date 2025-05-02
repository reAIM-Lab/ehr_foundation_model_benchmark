import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import json
from time import perf_counter as pc

from ehr_foundation_model_benchmark.tools.path import files
from ehr_foundation_model_benchmark.tools.mappings import (
    get_conversions,
    get_copy_majority_units,
    get_one_unit_lab,
    # compute_most_common_units,
    # get_one_unit_and_missing_lab,
    convert_mappings_to_id,
    # get_rare_units_labs,
    simplify_equivalent_units,
)

### PREREQUISITES
### The file measurement measurement_unit_counts.csv should have been created before and be in the current working directory

# Demo mode does not process all the labs, only one for each pipeline step to check the pipeline runs
demo = False
one_file = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return super().default(obj)

def report_harmonized(data, name):
    print(
        f"Harmonized units percentage {name}",
        100 - data["harmonized_value_as_number"].isnull().sum() * 100 / len(data),
        "%",
    ) 


def process_file(file):
    report_data = []

    print("Loading", file)
    data = pd.read_parquet(file)

    # new columns for the output
    data["harmonized_value_as_number"] = None
    data["harmonized_unit_concept_id"] = None

    # EQUIVALENT UNITS
    print("Simplifying units")
    s1 = pc()
    data, count = simplify_equivalent_units(data)
    s2 = pc()
    print("End simplifying units")
    report_data.append(("Simplifying units", count, count / len(data) * 100, s2 - s1))
    # exit()

    # UNDEFINED LABS
    print(len(data))
    s1 = pc()
    cdt = data.measurement_concept_id != 0
    count = np.count_nonzero(cdt)
    len_data_before = len(data)
    # print(report_data)
    # exit()
    if not demo:
        data = data.loc[cdt].copy()  # check 130 millions
    s2 = pc()
    report_data.append(("Undefined labs", len_data_before - count, 100 - count / len_data_before * 100, s2 - s1)) # then denominator change caution


    # SINGLE UNIT LABS
    n_rows_sul = 0
    s1 = pc()
    for measurement_id in tqdm(get_one_unit_lab(), desc="Single unit labs"):
        cdt = data["measurement_concept_id"] == measurement_id
        count = np.count_nonzero(cdt)
        if count > 0 and (~(data.loc[cdt, "unit_concept_id"].isnull())).any():
            n_rows_sul += count

            data.loc[cdt, "harmonized_value_as_number"] = data.loc[cdt, "value_as_number"]
            data.loc[cdt, "harmonized_unit_concept_id"] = data.loc[cdt, "unit_concept_id"]
            print("NA", count, data.loc[cdt, "unit_concept_id"].isnull().any())
            print("~NA", count,(~(data.loc[cdt, "unit_concept_id"].isnull())).any())
            print(len(data.loc[cdt, "unit_concept_id"]), data.loc[cdt, "unit_concept_id"].value_counts())

            if demo:
                break

    s2 = pc()

    report_harmonized(data, "after single unit labs")
    report_data.append(("Single unit labs", n_rows_sul, n_rows_sul / len(data) * 100, s2 - s1))

    # MULTI-UNIT LABS - CONVERSION
    s1 = pc()
    to_convert = get_conversions()
    mapping_functions_id = convert_mappings_to_id()
    n_rows_converted = 0
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

            n_rows_converted += np.count_nonzero(cdt)

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
    s2 = pc()
    report_harmonized(data, "after conversion")
    report_data.append(("Converted rows", n_rows_converted, n_rows_converted / len(data) * 100, s2 - s1))

    # MULTI-UNIT LABS - COPY MAJORITY UNITS
    # it is modelled as a conversion from target_unit to target_unit
    # already_done = []
    s1 = pc()
    to_copy = get_copy_majority_units()
    n_rows_copied = 0
    for measurement_id, from_unit_id, to_unit_id in tqdm(
        to_copy, desc="Multi unit labs - majority units"
    ):
        cdt = (data["unit_concept_id"] == from_unit_id) & (
            data["measurement_concept_id"] == measurement_id
        )
        n_rows_copied += np.count_nonzero(cdt)
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

    s2 = pc()
    report_harmonized(data, "after majority units copy")
    report_data.append(("Copied rows", n_rows_copied, n_rows_copied / len(data) * 100, s2 - s1))

    print(report_data)
    with open(file.replace(".snappy.parquet", "-processed_stats.csv"), 'w') as f:
        json.dump(report_data, f, cls=NumpyEncoder)

    # # fill in unit match type: majority, minority, nan (acceptable, unacceptable)
    # data['unit_match_type'] = 'to_fill'
    # data['harmonized_unit_concept_id'].fillna(0, inplace=True)
    # data.loc[(data['harmonized_unit_concept_id'] == data['majority_unit_id']) & (data['majority_unit_id'] == 0), 'unit_match_type'] = 'Mapped to majority unit (nan)'
    # data.loc[(data['harmonized_unit_concept_id'] == data['majority_unit_id']) & (data['majority_unit_id'] != 0), 'unit_match_type'] = 'Mapped to majority unit (non-nan)'
    # data.loc[(data['harmonized_unit_concept_id'] != data['majority_unit_id']) & (data['majority_unit_id'] != 0), 'unit_match_type'] = 'Mapped to minority unit'
    # data.loc[(data['harmonized_unit_concept_id'] == 0) & (data['majority_unit_id'] != 0), 'unit_match_type'] = 'Unmapped nan'
    
    # # fill in concept id type
    # data['value_match_type'] = 'to_fill'
    # data.loc[~(data['harmonized_value_as_number'].isna()), 'value_match_type'] = 'Numerical value'
    # data.loc[(data['harmonized_value_as_number'].isna()) & ~((data['value_as_concept_id'].isna())), 'value_match_type'] = 'Concept value (no number)'
    # data.loc[(data['harmonized_value_as_number'].isna()) & ((data['value_as_concept_id'].isna())), 'value_match_type'] = 'Nan concept and number'

    # # added: is the original unit concept ID nan/0? 
    # data['original_unit_type'] = 'Non-Nan'
    # data.loc[((data['unit_concept_id'].isna()) | (data['unit_concept_id']==0)), 'original_unit_type'] = 'Nan'
    
    # counts = data.groupby(by=["unit_match_type", "value_match_type", "original_unit_type"]).size().reset_index()
    
    # counts.to_csv(file.replace(".snappy.parquet", "-processed_unit_value_counts.csv"))

    report_harmonized(data, "at the end of the pipeline")

    if not demo:
        print("save")
        data.to_parquet(file.replace(".snappy.parquet", "-harmonized-v5.snappy.parquet"))
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

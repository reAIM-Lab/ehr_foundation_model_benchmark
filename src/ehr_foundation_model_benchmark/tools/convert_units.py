# import pandas as pd
import cudf as pd
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
    load_data
)

### PREREQUISITES
### The file measurement measurement_unit_counts.csv should have been created before and be in the current working directory

# Demo mode does not process all the labs, only one for each pipeline step to check the pipeline runs
demo = False


def report_harmonized(data, name):
    print(
        f"Harmonized units percentage {name}",
        100 - data["harmonized_value_as_number"].isnull().sum() * 100 / len(data),
        "%",
    )  # or use mean https://stackoverflow.com/questions/51070985/find-out-the-percentage-of-missing-values-in-each-column-in-the-given-dataset


###############
# TODO: parallel processing for each file with multiprocessing - very long otherwise (5-10 hours)
###############
def process_file(file):
    print("Loading", file)
    data = pd.read_parquet(file)

    # print(data.dtypes)
    # print(load_data().dtypes)
    # exit()

    data["harmonized_value_as_number"] = None
    data["harmonized_unit_concept_id"] = None

    most_common_units = compute_most_common_units()

    # EQUIVALENT UNITS
    print("Simplifying units")
    data = simplify_equivalent_units(data)
    print("End simplifying unis")

    # UNDEFINED LABS
    print(len(data))
    if not demo:
        # data = data.loc[data.measurement_concept_id != 0]  # check 130 millions
        pass

    # SINGLE UNIT LABS
    for measurement_id in tqdm(get_one_unit_lab()[48:], desc="Single unit labs"):
        cdt = (data["measurement_concept_id"] == measurement_id.get()) & \
            (~data["value_as_number"].isnull())
        
        print(cdt.sum())
        to_replace = data.loc[cdt, "value_as_number"]
        to_replace_concept = data.loc[cdt, "unit_concept_id"]
        
        if cdt.sum()==1:
            print(to_replace)
            to_replace = to_replace.values[0]
            to_replace_concept = to_replace_concept.values[0]

        data.loc[cdt, "harmonized_value_as_number"] = to_replace
        data.loc[cdt, "harmonized_unit_concept_id"] = to_replace_concept

        if demo:
            break

    report_harmonized(data, "after single unit labs")

    # ONE UNIT AND MISSING LABS
    for measurement_id in tqdm(
        get_one_unit_and_missing_lab(), desc="Single unit + missing labs"
    ):
        cdt = (data["measurement_concept_id"] == measurement_id) & \
            (~data["value_as_number"].isnull())
        
        print(cdt.sum())

        to_replace = data.loc[cdt, "value_as_number"]
        
        if cdt.sum()==1:
            to_replace = to_replace.values[0]

        data.loc[cdt, "harmonized_value_as_number"] = to_replace
        data.loc[cdt, "harmonized_unit_concept_id"] = int(most_common_units[measurement_id])

        if demo:
            break

    report_harmonized(data, "after single unit and missing labs")

    # MULTI-UNIT LABS
    to_convert = get_conversions()
    mapping_functions_id = convert_mappings_to_id()
    for measurement_id, from_unit_id, to_unit_id in tqdm(
        to_convert, desc="Multi unit labs"
    ):
        measurement_id = int(measurement_id)
        from_unit_id = int(from_unit_id)
        to_unit_id = int(to_unit_id)
        cdt2 = (from_unit_id, to_unit_id, measurement_id) in mapping_functions_id
        if (from_unit_id, to_unit_id, None) in mapping_functions_id or cdt2:
            cdt = (data["unit_concept_id"] == from_unit_id) & (
                data["measurement_concept_id"] == measurement_id
            ) & (~data['value_as_number'].isnull())
            if cdt2:
                print("Applying additional filter")
                cdt = cdt & (data["measurement_concept_id"] == measurement_id)

            print(
                "Converting",
                measurement_id,
                from_unit_id,
                to_unit_id,
                # np.count_nonzero(cdt),
                cdt.sum()
            )
            # count can be 0 because only one file here and not everything is loaded like for the measurement_unit_counts

            if cdt2:
                mapping_fun = mapping_functions_id[
                    (from_unit_id, to_unit_id, measurement_id)
                ]
            else:
                mapping_fun = mapping_functions_id[(from_unit_id, to_unit_id, None)]

            # mapping_fun = lambda x: float(mapping_fun(x))

            data.loc[cdt, "harmonized_value_as_number"] = data.loc[
                cdt, "value_as_number"
            ].map(mapping_fun).to_numpy() # when some nulls in the columns, exclude them?
            # if not null, do not convert => exclude before??
            data.loc[cdt, "harmonized_unit_concept_id"] = to_unit_id

        if demo:
            break

    report_harmonized(data, "after conversion")

    # RARE UNITS
    # for rare units, convert to nan (0) in the unit_concept_id

    # print(data.columns)
    rare_units = get_rare_units_labs()
    for measurement_id, unit_id in tqdm(rare_units, desc="Rare units"):
        cdt = (
            (data["unit_concept_id"] == unit_id)
            & (data["measurement_concept_id"] == measurement_id)
            & (data["harmonized_value_as_number"].isnull())
        )  # not already converted

        data.loc[cdt, "unit_concept_id"] = 0
        # data.loc[cdt, "unit_concept_name"] = "No matching concept"

        if demo:
            break

    ###########################################
    # FURTHER PROCESSING NEEDED TO RESOLVE NANS
    ###########################################
    # TODO
    ###########################################

    report_harmonized(data, "at the end of the pipeline")

    if not demo:
        print("save")
        data.to_parquet(file.replace(".snappy.parquet", "-harmonized.snappy.parquet"))
        print("end")

    
if __name__ == "__main__":

    max_processes = 1 # 16 makes the server crash

    # print(len(files))
    # exit()

    if True:
        files = [files[0]]#, files[1]]

    # Create a pool of workers (with a maximum of 4 processes)
    with mp.get_context("spawn").Pool(processes=max_processes) as pool:
        # Map the process_file function to each file path in the list
        results = pool.map(process_file, files)

    # Print results or handle them as needed
    for result in results:
        print(f"File processed with result: {result}")
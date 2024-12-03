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
    file = file.replace(".snappy.parquet", "-harmonized.snappy.parquet")
    print("Loading", file)
    df = pd.read_parquet(file)

    # data["harmonized_value_as_number"] = None
    # data["harmonized_unit_concept_id"] = None

    most_common_units = compute_most_common_units()

    # EQUIVALENT UNITS
    # print("Simplifying units")
    # data = simplify_equivalent_units(data)
    # print("End simplifying unis")

    # UNDEFINED LABS
    # print(len(data))
    # if not demo:
    #     data = data.loc[data.measurement_concept_id != 0]  # check 130 millions

    # # SINGLE UNIT LABS
    # for measurement_id in tqdm(get_one_unit_lab(), desc="Single unit labs"):
    #     cdt = data["measurement_concept_id"] == measurement_id

    #     data.loc[cdt, "harmonized_value_as_number"] = data.loc[cdt, "value_as_number"]
    #     data.loc[cdt, "harmonized_unit_concept_id"] = data.loc[cdt, "unit_concept_id"]

    #     if demo:
    #         break

    # report_harmonized(data, "after single unit labs")

    # # ONE UNIT AND MISSING LABS
    # for measurement_id in tqdm(
    #     get_one_unit_and_missing_lab(), desc="Single unit + missing labs"
    # ):
    #     cdt = data["measurement_concept_id"] == measurement_id

    #     data.loc[cdt, "harmonized_value_as_number"] = data.loc[cdt, "value_as_number"]
    #     data.loc[cdt, "harmonized_unit_concept_id"] = most_common_units[measurement_id]

    #     if demo:
    #         break

    # report_harmonized(data, "after single unit and missing labs")

    # MULTI-UNIT LABS
    to_convert = get_conversions()
    mapping_functions_id = convert_mappings_to_id()
    already_done = []
    for measurement_id, from_unit_id, to_unit_id in tqdm(
        to_convert, desc="Multi unit labs"
    ):
        cdt2 = (from_unit_id, to_unit_id, measurement_id) in mapping_functions_id
        if (from_unit_id, to_unit_id, None) in mapping_functions_id or cdt2:
            if not (measurement_id, to_unit_id) in already_done:
                cdt = (df["unit_concept_id"] == to_unit_id) & (
                    df["measurement_concept_id"] == measurement_id
                )
                # if cdt2:
                #     print("Applying additional filter")
                #     cdt = cdt & (df["measurement_concept_id"] == measurement_id)

                print(
                    "Converting",
                    measurement_id,
                    to_unit_id,
                    to_unit_id,
                    np.count_nonzero(cdt),
                )
                # count can be 0 because only one file here and not everything is loaded like for the measurement_unit_counts

                # if cdt2:
                #     mapping_fun = mapping_functions_id[
                #         (from_unit_id, to_unit_id, measurement_id)
                #     ]
                # else:
                #     mapping_fun = mapping_functions_id[(from_unit_id, to_unit_id, None)]

                mapping_fun = lambda x: x

                df.loc[cdt, "harmonized_value_as_number"] = df.loc[
                    cdt, "value_as_number"
                ].apply(mapping_fun)
                df.loc[cdt, "harmonized_unit_concept_id"] = to_unit_id

        # if demo:
        #     break

    # cdt = ((~df['value_as_number'].isnull()) | (~df['value_as_concept_id'].isnull())) & (df['harmonized_unit_concept_id'].isnull())
    # # cdt2 = cdt & (df['unit_concept_id'].isnull()) # is null vs 0?
    # # print("Non harmonized with value as number or as concept", compute_percentage(np.count_nonzero(cdt), df))
    # # print("Non harmonized with value as number or as concept AND with NaN unit", compute_percentage(np.count_nonzero(cdt2), df))

    # result = df[cdt].groupby('measurement_concept_id').agg(
    #     unique_unit_count=('unit_concept_id', 'nunique'),
    #     total_measurements=('measurement_id', 'count')
    # ).reset_index()

    # result = result.sort_values(by='total_measurements', ascending=False)


    # # Display or print the result
    # print(result)
    # print(result['unique_unit_count'].value_counts())
    # # result.to_csv('agg.csv')
    # # result_cdt = (result['unique_unit_count'] > 1)
    # # print(result[result_cdt]['total_measurements'].sum())
    # result_cdt = (result['unique_unit_count'] == 1)
    # print(result[result_cdt]['total_measurements'].sum())
    # # result_cdt = (result['unique_unit_count'] == 0) # why it does not match Non harmonized with value as number or as concept AND with NaN unit
    # # print(result[result_cdt]['total_measurements'].sum())

    # # for measurement_concept_id in result[result_cdt]:
    # #     # basically rerun convert single labs and labs + nans on the harmonized: did it replace in place?
    # #     # take measurement and convert, only nan have changed? 
    # #     pass

    # report_harmonized(df, "after conversion")

    # RARE UNITS
    # for rare units, convert to nan (0) in the unit_concept_id
    # rare_units = get_rare_units_labs()
    # for measurement_id, unit_id in tqdm(rare_units, desc="Rare units"):
    #     cdt = (
    #         (data["unit_concept_id"] == unit_id)
    #         & (data["measurement_concept_id"] == measurement_id)
    #         & (data["harmonized_value_as_number"].isnull())
    #     )  # not already converted

    #     data.loc[cdt, "unit_concept_id"] = 0
        
    #     # column does not exist in data in fact, no need
    #     data.loc[cdt, "unit_concept_name"] = "No matching concept"

    #     if demo:
    #         break

    ###########################################
    # FURTHER PROCESSING NEEDED TO RESOLVE NANS
    ###########################################
    # TODO
    ###########################################

    report_harmonized(df, "at the end of the pipeline")

    if not demo:
        print("save")
        df.to_parquet(file.replace(".snappy.parquet", "-harmonized-2.snappy.parquet"))
        print("end")

    
if __name__ == "__main__":

    max_processes = 2 # 16 makes the server crash

    # print(len(files))
    # exit()

    if demo:
        files = [files[0]]#, files[1]]

    # files = [files[3], files[5], files[9], files[11]]

    # Create a pool of workers (with a maximum of 4 processes)
    with mp.get_context("spawn").Pool(processes=max_processes) as pool:
        # Map the process_file function to each file path in the list
        results = pool.map(process_file, files)

    # Print results or handle them as needed
    for result in results:
        print(f"File processed with result: {result}")
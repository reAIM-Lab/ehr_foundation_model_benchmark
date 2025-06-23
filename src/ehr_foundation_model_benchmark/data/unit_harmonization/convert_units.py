import functools
import os
import glob
import json
from time import perf_counter as pc

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from ehr_foundation_model_benchmark.data.unit_harmonization.mappings import (
    get_conversions,
    get_copy_majority_units,
    get_one_unit_lab,
    convert_mappings_to_id,
    simplify_equivalent_units,
)


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


def process_file(
        file,
        dry_run: bool = False
):
    report_data = []

    print("Loading", file)
    data = pd.read_parquet(file)

    # New columns for the output
    data["harmonized_value_as_number"] = None
    data["harmonized_unit_concept_id"] = None

    # Simplify equivalent units
    print("Simplifying units")
    s1 = pc()
    data, count = simplify_equivalent_units(data)
    s2 = pc()
    print("End simplifying units")
    report_data.append(("Simplifying units", count, count / len(data) * 100, s2 - s1))

    # Filter undefined labs
    print(len(data))
    s1 = pc()
    cdt = data.measurement_concept_id != 0
    count = np.count_nonzero(cdt)
    len_data_before = len(data)

    if not dry_run:
        data = data.loc[cdt].copy()
    s2 = pc()
    report_data.append(("Undefined labs", len_data_before - count, 100 - count / len_data_before * 100, s2 - s1))

    # Single unit labs
    n_rows_sul = 0
    s1 = pc()
    for measurement_id in tqdm(get_one_unit_lab(), desc="Single unit labs"):
        cdt = data["measurement_concept_id"] == measurement_id
        count = np.count_nonzero(cdt)
        if count > 0 and (~(data.loc[cdt, "unit_concept_id"].isnull())).any():
            n_rows_sul += count

            data.loc[cdt, "harmonized_value_as_number"] = data.loc[cdt, "value_as_number"]
            data.loc[cdt, "harmonized_unit_concept_id"] = data.loc[cdt, "unit_concept_id"]

            if dry_run:
                break

    s2 = pc()
    report_harmonized(data, "after single unit labs")
    report_data.append(("Single unit labs", n_rows_sul, n_rows_sul / len(data) * 100, s2 - s1))

    # Multi-unit labs - conversion
    s1 = pc()
    to_convert = get_conversions()
    mapping_functions_id = convert_mappings_to_id()
    n_rows_converted = 0
    for measurement_id, from_unit_id, to_unit_id in tqdm(
            to_convert, desc="Multi unit labs - conversion"
    ):
        cdt2 = (from_unit_id, to_unit_id, measurement_id) in mapping_functions_id
        if (from_unit_id, to_unit_id, None) in mapping_functions_id or cdt2:
            cdt = (data["unit_concept_id"] == from_unit_id) & (
                    data["measurement_concept_id"] == measurement_id
            )

            n_rows_converted += np.count_nonzero(cdt)

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

            if dry_run:
                break
    s2 = pc()
    report_harmonized(data, "after conversion")
    report_data.append(("Converted rows", n_rows_converted, n_rows_converted / len(data) * 100, s2 - s1))

    # Multi-unit labs - copy majority units
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
        mapping_fun = lambda x: x

        data.loc[cdt, "harmonized_value_as_number"] = data.loc[
            cdt, "value_as_number"
        ].apply(mapping_fun)
        data.loc[cdt, "harmonized_unit_concept_id"] = to_unit_id

        if dry_run:
            break

    s2 = pc()
    report_harmonized(data, "after majority units copy")
    report_data.append(("Copied rows", n_rows_copied, n_rows_copied / len(data) * 100, s2 - s1))

    print(report_data)
    with open(file.replace(".snappy.parquet", "-processed_stats.csv"), 'w') as f:
        json.dump(report_data, f, cls=NumpyEncoder)

    report_harmonized(data, "at the end of the pipeline")

    if not dry_run:
        print("save")
        data.to_parquet(file.replace(".snappy.parquet", "-harmonized-v5.snappy.parquet"))
        print("end")


if __name__ == "__main__":
    from argparse import ArgumentParser
    argparser = ArgumentParser("unit conversion")
    argparser.add_argument(
        "--source_measurement_dir",
        dest="source_measurement_dir",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--demo",
        dest="demo",
        action="store_true"
    )
    argparser.add_argument(
        "--dry_run",
        dest="dry_run",
        action="store_true"
    )
    argparser.add_argument(
        "--one_file",
        dest="one_file",
        action="store_true"
    )
    argparser.add_argument(
        "--max_processes",
        dest="max_processes",
        action="store",
        type=int,
        default=1,
        required=False,
    )
    args = argparser.parse_args()

    files = glob.glob(os.path.join(args.source_measurement_dir, "*.parquet"))

    # The dry_run mode does not process all the labs, only one for each pipeline step to check the pipeline runs
    if args.demo:
        files = [files[0], files[1]]

    if args.one_file:
        files = [files[0]]

    with mp.get_context("spawn").Pool(processes=args.max_processes) as pool:
        results = pool.map(
            functools.partial(process_file, dry_run=args.dry_run), files
        )

    for result in results:
        print(f"File processed with result: {result}")
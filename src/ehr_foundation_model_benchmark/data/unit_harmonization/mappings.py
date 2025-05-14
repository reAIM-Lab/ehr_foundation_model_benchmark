import pandas as pd
from tqdm import tqdm
import warnings

import json
from ehr_foundation_model_benchmark.tools.path import concepts_path

identity = lambda x: x

# third column is measurement id when it is lab specific
# all identities should be in equivalent?
mapping_functions = {
    # identity mappings
    ("billion per liter", "billion per liter", None): identity,
    ("calculated", "ratio", None): identity,
    ("Ehrlich unit per deciliter", "milligram per deciliter", None): identity,
    ("femtoliter", "percent", None): identity,
    ("gram per deciliter", "ratio", None): identity,
    ("international unit per liter", "milligram per deciliter", None): identity,
    (
        "micro-international unit per milliliter",
        "milli-international unit per liter",
        None,
    ): identity,
    ("milliequivalent per liter", "millimole per liter", None): identity,
    ("milligram", "milligram per deciliter", None): identity,
    ("milligram per deciliter", "Ehrlich unit", None): identity,
    ("millimeter", "millimeter per hour", None): identity,
    ("millimeter mercury column", "millimeter", None): identity,
    ("millimeter mercury column", "percent", None): identity,
    ("percent", "gram per deciliter", None): identity,
    ("per 100 white blood cells", "percent", None): identity,
    ("phot", "pH", None): identity,
    ("picogram", "femtoliter", None): identity,
    # Non-Identity Mappings
    ("centimeter", "inch (US)", None): lambda x: x * 0.393701,
    ("degree Celsius", "degree Fahrenheit", None): lambda x: x * 9 / 5 + 32,
    ("gram per deciliter", "milligram per deciliter", None): lambda x: x * 1000,
    ("gram per liter", "milligram per deciliter", None): lambda x: x * 100,
    (
        "international unit per deciliter",
        "international unit per liter",
        None,
    ): lambda x: x
    * 10,
    ("kilo-international unit per liter", "billion per liter", None): lambda x: x * 1e6,
    ("kilogram", "ounce (avoirdupois)", None): lambda x: x * 35.274,
    ("microgram per deciliter", "milligram per milliliter", None): lambda x: x * 0.1,
    ("milligram per deciliter", "gram per deciliter", None): lambda x: x / 1000,
    ("milligram per liter", "microgram per deciliter", None): lambda x: x * 100,
    ("milligram per liter", "milligram per deciliter", None): lambda x: x * 0.1,
    ("millimole per liter", "milligram per deciliter", 3027114): lambda x: x * 38.67,
    ("millimole per liter", "milligram per deciliter", 3050068): lambda x: x * 0.2496,
    (
        "milli-international unit per liter",
        "micro-international unit per milliliter",
        None,
    ): lambda x: x
    * 1000,
    (
        "milli-international unit per milliliter",
        "micro-international unit per milliliter",
        None,
    ): lambda x: x
    * 1000,
    ("nanogram per deciliter", "nanogram per milliliter", None): lambda x: x * 0.01,
    ("nanogram per milliliter", "milligram per deciliter", None): lambda x: x * 0.0001,
    ("nanogram per milliliter", "milligram per liter", None): lambda x: x * 0.001,
    ("nanogram per milliliter", "nanogram per deciliter", None): lambda x: x * 100,
    ("ounce (avoirdupois)", "pound (US)", None): lambda x: x / 16,
    ("per cubic millimeter", "billion per liter", None): lambda x: x * 0.001,
    ("per microliter", "billion per liter", None): lambda x: x * 0.001,
    ("pound (US)", "ounce (avoirdupois)", None): lambda x: x * 16,
}
# check no unit from the first column of conversion mappings is in the first column of equivalent mappings
# check no unit from the second column of conversion mappings is in the first column of equivalent mappings


def convert_mappings_to_id():
    mapping_functions_id = {}
    for (from_unit, to_unit, filters), val in mapping_functions.items():
        mapping_functions_id[
            (convert_to_id(from_unit), convert_to_id(to_unit), filters)
        ] = val
    return mapping_functions_id


# how was this list obtained?
# could it be just in mapping_functions with identity to simplify?
# most common units use renamed units? yes
mappings_equivalent_units = {
    "arbitrary unit per liter": "international unit per liter",
    "arbitrary unit per milliliter": "international unit per milliliter",
    "cells per high power field": "per high power field",
    "cells per microliter": "per microliter",
    "counts per minute": "per minute",
    "inch (international)": "inch (US)",
    "microgram per liter": "nanogram per milliliter",
    "microgram per milliliter": "milligram per liter",
    "milliliter per minute": "milliliter per minute per 1.73 square meter",
    "Milligram per day": "milligram per 24 hours",
    "million per microliter": "trillion per liter",
    "nanomole per milliliter": "micromole per liter",
    "picogram per milliliter": "nanogram per liter",
    "thousand per cubic millimeter": "billion per liter",
    "thousand per microliter": "billion per liter",
    "unit per liter": "international unit per liter",
}


# to avoid reloading the file every time
concepts_df = None


def convert_to_id(name):
    global concepts_df

    # per minute has several ids, depending on the vocabulary, two are units
    if concepts_df is None:
        concepts_df = pd.read_parquet(concepts_path)  # [['concept_id', 'concept_name']]
        concepts_df = concepts_df[
            (
                (concepts_df["domain_id"] == "Unit")
                & (concepts_df["concept_class_id"] == "Unit")
            )
        ]

    try:
        matches = concepts_df[concepts_df["concept_name"] == name]
        assert len(matches) == 1
    except:
        # warnings.warn(f"Multiple concepts have the same name '{name}', taking the valid one")
        concepts_df_valid = concepts_df[concepts_df["invalid_reason"].isnull()]

        matches = concepts_df_valid[concepts_df_valid["concept_name"] == name]
        assert len(matches) == 1

    return matches["concept_id"].values[0]


def simplify_equivalent_units(df_labs):
    # should be applied to the whole dataset in addition to aggregated
    if "unit_concept_name" in df_labs.columns:
        df_labs["original_unit_concept_name"] = df_labs["unit_concept_name"].copy()
        df_labs["unit_concept_name"] = df_labs["unit_concept_name"].replace(
            mappings_equivalent_units
        )
    mappings_equivalent_units_id = {}
    for key, item in mappings_equivalent_units.items():
        mappings_equivalent_units_id[convert_to_id(key)] = convert_to_id(item)
    df_labs["original_unit_concept_id"] = df_labs["unit_concept_id"].copy()
    df_labs["unit_concept_id"] = df_labs["unit_concept_id"].replace(
        mappings_equivalent_units_id
    )
    replaced_count = (df_labs["original_unit_concept_id"] != df_labs["unit_concept_id"]).sum()
    # print(f"Number of rows replaced: {replaced_count}")

    return df_labs, replaced_count


def load_data():
    df_labs = pd.read_csv("measurement_unit_counts.csv")
    df_labs.drop("Unnamed: 0", axis=1, inplace=True)

    df_labs.fillna({"unit_concept_id": 0}, inplace=True)
    df_labs.fillna({"unit_concept_name": "No matching concept"}, inplace=True)

    df_labs, _ = simplify_equivalent_units(df_labs)
    # how will it impact the statistics ?

    df_labs = (
        df_labs.groupby(
            [
                "measurement_concept_id",
                "measurement_name",
                "unit_concept_id",
                "unit_concept_name",
            ]
        )
        .sum()
        .reset_index()
    )
    return df_labs


def get_one_unit_lab():
    df_labs = load_data()

    labs_per_unit = df_labs.groupby("measurement_concept_id").count()["unit_concept_id"] # should not be NaN
    one_lab_per_unit = labs_per_unit.loc[labs_per_unit == 1].index
    return one_lab_per_unit.values


def get_one_unit_and_missing_lab():  # retour index of labs
    df_labs = load_data()

    single_unit_plus_missing_labs = []
    for i in df_labs.measurement_concept_id.unique():
        temp_labs = df_labs.set_index("measurement_concept_id").loc[i]
        if (len(temp_labs) == 2) and (0 in list(temp_labs["unit_concept_id"])):
            single_unit_plus_missing_labs.append(i)

    return single_unit_plus_missing_labs


def get_rare_units_labs():
    df_labs = load_data()

    # should we apply the transformation to the original dataset
    # or should we recompute the counts after the first groups of conversion
    # potential issue: partial stats per file?
    # use cudf / dask to speed up?

    rare_units = []
    for i in df_labs["measurement_concept_id"].unique():
        temp_multi_labs = df_labs.loc[df_labs["measurement_concept_id"] == i]
        temp_multi_labs = (
            temp_multi_labs.groupby("unit_concept_id").sum()["counts"]
            / temp_multi_labs["counts"].sum()
        ).reset_index()
        if (
            temp_multi_labs["counts"].min() < 0.001 and len(temp_multi_labs) >= 2
        ):  # if two units not nan
            for idx, row in temp_multi_labs.iterrows():
                if row["counts"] < 0.001:
                    rare_units.append((i, idx))

    return rare_units


def compute_most_common_units():
    # returns: dict

    df_labs = load_data()

    dict_most_common_id = {}
    for i in df_labs["measurement_concept_id"].unique():
        temp_labs = df_labs.loc[df_labs["measurement_concept_id"] == i]
        units = temp_labs.sort_values("counts", ascending=False)

        units = units[~(units.unit_concept_id == 0)]
        try:
            dict_most_common_id[i] = units["unit_concept_id"].values[0]
        except:
            dict_most_common_id[i] = 0  # not unit for all measurements

    return dict_most_common_id


def get_conversions():
    df_labs = load_data()
    to_convert = []
    most_common = compute_most_common_units()

    for i in tqdm(df_labs["measurement_concept_id"].unique(), "Conversions"):
        temp_multi_labs = df_labs.loc[df_labs["measurement_concept_id"] == i]
        if len(temp_multi_labs) >= 2:  # two different units, include nan and real unit
            to_unit = most_common[i]
            for unit in temp_multi_labs["unit_concept_id"]:
                if unit != 0 and to_unit != unit:
                    to_convert.append((i, unit, to_unit))

    return to_convert

def get_copy_majority_units():
    df_labs = load_data()
    to_copy = []
    most_common = compute_most_common_units()

    for i in tqdm(df_labs["measurement_concept_id"].unique(), "Conversions"):
        temp_multi_labs = df_labs.loc[df_labs["measurement_concept_id"] == i]
        if len(temp_multi_labs) >= 2:  # two different units, include nan and real unit
            to_unit = most_common[i]
            to_copy.append((i, to_unit, to_unit))
            # to unit can be nan?

    return to_copy



def get_filter(df, mapping_key):
    return (df["unit_concept_name"] == mapping_key[0]) & (
        df["most_common_unit"] == mapping_key[1]
    )


if __name__ == "__main__":
    # print(remove_rare[3007461])
    most_common_units = compute_most_common_units()
    most_common_units = {int(key): value for key, value in most_common_units.items()}

    print(most_common_units)

    with open('most_common_units.json', 'w') as f:
        json.dump(most_common_units, f)
    # print(convert_to_id('per minute'))
    # print(load_data())
    # print(get_rare_units_labs())
    # print(get_conversions())

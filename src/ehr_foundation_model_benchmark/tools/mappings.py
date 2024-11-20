import pandas as pd
from tqdm import tqdm
import warnings

from ehr_foundation_model_benchmark.tools.path import concepts_path

mapping_functions = \
{
    ("degree Celsius", "degree Fahrenheit"): lambda x: x*9/5 + 32,
    ("pound (US)", "ounce (avoirdupois)"): lambda x: x*16,
    ("kilogram", "ounce (avoirdupois)"): lambda x: x*35.274,
    ("centimeter", "inch (US)"): lambda x: x*0.393701,
    ("nanogram per milliliter", "milligram per liter"): lambda x: x*0.001,
    ("milligram per liter", "milligram per deciliter"): lambda x: x*0.1,
    ("per microliter", "billion per liter"): lambda x: x*0.001,
    ("milligram per liter", "milligram per deciliter"): lambda x: x*0.1,
    # continue mappings based on table

}

def convert_mappings_to_id():
    mapping_functions_id = {}
    for (from_unit, to_unit), val in mapping_functions.items():
        mapping_functions_id[(convert_to_id(from_unit), convert_to_id(to_unit))] = val
    return mapping_functions_id


# how was this list obtained?
# could it be just in mapping_functions with identity to simplify?
mappings_equivalent_units = {'counts per minute':'per minute',
           'thousand per microliter': 'billion per liter', 
           'thousand per cubic millimeter': 'billion per liter',
           'million per microliter': 'trillion per liter', 
           'unit per liter': 'international unit per liter',
            'arbitrary unit per liter': 'international unit per liter',
           'arbitrary unit per milliliter': 'international unit per milliliter',
            'inch (international)':'inch (US)',
           'milliliter per minute':'milliliter per minute per 1.73 square meter',
           'cells per microliter': 'per microliter',
           'cells per high power field':'per high power field',
           'microgram per liter':'nanogram per milliliter',
           'picogram per milliliter':'nanogram per liter',
           'nanomole per milliliter':'micromole per liter',
           'microgram per milliliter':'milligram per liter',
           'Milligram per day':'milligram per 24 hours'}

# to avoid reloading the file every time
concepts_df = None

def convert_to_id(name):
    global concepts_df

    # per minute has several ids, depending on the vocabulary, two are units
    if concepts_df is None:
        concepts_df = pd.read_parquet(concepts_path)#[['concept_id', 'concept_name']]
        concepts_df = concepts_df[((concepts_df['domain_id'] == "Unit") & (concepts_df['concept_class_id'] == "Unit"))]
    
    try:
        matches = concepts_df[concepts_df['concept_name'] == name]
        assert len(matches) == 1
    except:
        # warnings.warn(f"Multiple concepts have the same name '{name}', taking the valid one")
        concepts_df_valid = concepts_df[concepts_df['invalid_reason'].isnull()]

        matches = concepts_df_valid[concepts_df_valid['concept_name'] == name]
        assert len(matches) == 1

    return matches['concept_id'].values[0]


def simplify_equivalent_units(df_labs):
    df_labs['unit_concept_name'] = df_labs['unit_concept_name'].replace(mappings_equivalent_units)
    mappings_equivalent_units_id = {}
    for key, item in (mappings_equivalent_units.items()):
        mappings_equivalent_units_id[convert_to_id(key)] = convert_to_id(item)
    df_labs['unit_concept_id'] = df_labs['unit_concept_id'].replace(mappings_equivalent_units_id)
    return df_labs


def load_data():
    df_labs = pd.read_csv('measurement_unit_counts.csv')
    df_labs.drop('Unnamed: 0', axis = 1, inplace=True)

    df_labs.fillna({'unit_concept_id': 0}, inplace=True)
    df_labs.fillna({'unit_concept_name': 'No matching concept'}, inplace=True)

    df_labs = simplify_equivalent_units(df_labs)

    df_labs = df_labs.groupby(['measurement_concept_id', 'measurement_name','unit_concept_id', 'unit_concept_name']).sum().reset_index()
    return df_labs


def get_one_unit_lab():
    df_labs = load_data()
    labs_per_unit = df_labs.groupby('measurement_concept_id').count()['unit_concept_id']
    one_lab_per_unit = labs_per_unit.loc[labs_per_unit == 1].index
    return one_lab_per_unit.values


def get_one_unit_and_missing_lab(): # retour index of labs
    df_labs = load_data()

    single_unit_plus_missing_labs = []
    for i in df_labs.measurement_concept_id.unique():
        temp_labs = df_labs.set_index('measurement_concept_id').loc[i]
        if (len(temp_labs) == 2) and (0 in list(temp_labs['unit_concept_id'])):
            single_unit_plus_missing_labs.append(i)

    return single_unit_plus_missing_labs

def get_rare_units_labs():
    df_labs = load_data()

    to_remove = []
    for i in df_labs['measurement_concept_id'].unique():
        temp_multi_labs = df_labs.loc[df_labs['measurement_concept_id']==i]
        temp_multi_labs = (temp_multi_labs.groupby('unit_concept_id').sum()['counts']/temp_multi_labs['counts'].sum()).reset_index()
        if temp_multi_labs['counts'].min() < 0.001 and len(temp_multi_labs) >= 3:
            # print(i)
            # print(temp_multi_labs)
            for idx, row in temp_multi_labs.iterrows():
                if row['counts'] < 0.001:
                    to_remove.append((i, idx))

    # idxes = []
    # for measurement_concept_id, unit_concept_id in to_remove:
    #     idxes.extend(df_labs[
    #         (df_labs['measurement_concept_id'] == measurement_concept_id) &
    #         (df_labs['unit_concept_id'] == unit_concept_id)
    #     ].index)

    return to_remove

    # delete the rows

    # print(len(idxes), idxes)
    # print(len(df_labs))
    # df_labs = df_labs.loc[~df_labs.index.isin(idxes)]
    # print(len(df_labs))

    return idxes

    

    # dict_units = compute_most_common_units()
    # # issue if most common unit is below 0.1%
    # print(dict_units[3011163])



# def get_unit_id(name): # group of equivalent unit_id
#     # unit_concept_name
#     # csv = pd.read_csv('measurement_unit_counts.csv')
#     # return csv[csv['unit_concept_name'] == name]['unit_concept_id'].values[0]
#     return convert_to_id(name)

def compute_most_common_units():
    # returns: dict

    df_labs = load_data()

    # df_labs['most_common_unit_id'] = None
    dict_most_common_id = {}
    for i in df_labs['measurement_concept_id'].unique():
        temp_labs = df_labs.loc[df_labs['measurement_concept_id'] == i]
        # dict_most_common[i] = temp_labs.loc[temp_labs['counts']==temp_labs['counts'].max(), 'unit_concept_name'].values[0]
        # units = temp_labs.loc[temp_labs['counts']==temp_labs['counts'].max(), 'unit_concept_id']
        units = temp_labs.sort_values('counts', ascending=False)
        # print(units)

        units = units[~(units.unit_concept_id == 0)]
        try:
            dict_most_common_id[i] = units['unit_concept_id'].values[0]
        except:
            dict_most_common_id[i] = 0 # not unit for all measurements
        # multi_unit_labs.loc[multi_unit_labs['measurement_concept_id']==i, 'most_common_unit'] = dict_most_common[i]
        # df_labs.loc[df_labs['measurement_concept_id']==i, 'most_common_unit_id'] = dict_most_common_id[i]

    # df_labs = df_labs.groupby('measurement_concept_id')#.set_index('measurement_concept_id')

    # return df_labs
    return dict_most_common_id

def get_conversions():
    df_labs = load_data()
    to_convert = []
    most_common = compute_most_common_units()
    for i in tqdm(df_labs['measurement_concept_id'].unique(), 'Conversions'):
        temp_multi_labs = df_labs.loc[df_labs['measurement_concept_id'] == i]
        if len(temp_multi_labs) >= 2: # two different units
            to_unit = most_common[i]
            for unit in temp_multi_labs['unit_concept_id']:
                if unit != 0 and to_unit != unit:
                    to_convert.append((i, unit, to_unit))

    # print(len(to_convert))
    return to_convert


def get_measurements(mapping_key):
    # df_labs = load_data()
    # select the multi units labs
    # csv['unit_concept_name'] == mapping_key[0]

    csv = pd.read_csv('to_convert.csv')

    # df_labs = load_data()

    # TBD

    return csv[(csv['unit_concept_name'] == mapping_key[0]) & (csv['most_common_unit'] == mapping_key[1])]['measurement_concept_id'].values


def get_filter(df, mapping_key):
    return (df['unit_concept_name'] == mapping_key[0]) & (df['most_common_unit'] == mapping_key[1])

# print(get_unit_id("degree Fahrenheit"))
# print(get_measurements(list(mapping_functions.keys())[0]))

if __name__ == '__main__':
    # print(remove_rare[3007461])
    # print(compute_most_common_units())
    # print(convert_to_id('per minute'))
    # print(load_data())
    # print(get_rare_units_labs())
    print(get_conversions())
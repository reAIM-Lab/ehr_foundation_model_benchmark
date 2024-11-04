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
}


# remove
remove_rare = {3000483: 'millimeter mercury column',
              3004239:'milligram per deciliter',
              3005131:'calculated',
              3006906:'millimole per liter',
              3007238:'percent',
              3007461: 'femtoliter', 3007461: 'percent',
              3011397:'Generic unit for indivisible thing',
              3019897: 'femtoliter',
              3007238: 'No matching concept', 3007238: 'gram per deciliter',
              3020509: 'gram per deciliter',
              3020876: 'milligram per deciliter',
              3021601: 'newton',
              3022621: 'milligram per deciliter', 3022621:'phot',
              3023599: 'picogram',
              3024629: 'Generic unit for indivisible thing',
              3027114: 'millimole per liter',
              3027597: 'international unit per liter',
              3039720: 'calculated',
              3050479: 'billion per liter',
              3020416: 'gram per deciliter',
              3013721: 'gram per deciliter'}


# CONVERSION
mappings_equivalent_units = {'counts per minute':'per minute',
           'thousand per microliter': 'billion per liter', 'thousand per cubic millimeter': 'billion per liter',
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

import pandas as pd

def load_data():
    df_labs = pd.read_csv('measurement_unit_counts.csv')
    df_labs.drop('Unnamed: 0', axis = 1, inplace=True)

    # df_labs.head()
    df_labs['unit_concept_id'].fillna(0, inplace=True)
    df_labs['unit_concept_name'].fillna('No matching concept', inplace=True)

    df_labs = df_labs.groupby(['measurement_concept_id', 'measurement_name','unit_concept_id', 'unit_concept_name']).sum().reset_index()
    return df_labs

def get_one_unit_lab():
    df_labs = load_data()
    labs_per_unit = df_labs.groupby('measurement_concept_id').count()['unit_concept_id']
    one_lab_per_unit = labs_per_unit.loc[labs_per_unit == 1].index
    return one_lab_per_unit.values

def get_unit_id(name): # group of equivalent unit_id
    # unit_concept_name
    csv = pd.read_csv('measurement_unit_counts.csv')
    return csv[csv['unit_concept_name'] == name]['unit_concept_id'].values[0]

def get_measurements(mapping_key):
    csv = pd.read_csv('to_convert.csv')
    return csv[(csv['unit_concept_name'] == mapping_key[0]) & (csv['most_common_unit'] == mapping_key[1])]['measurement_concept_id'].values

def get_filter(df, mapping_key):
    return (df['unit_concept_name'] == mapping_key[0]) & (df['most_common_unit'] == mapping_key[1])

# print(get_unit_id("degree Fahrenheit"))
# print(get_measurements(list(mapping_functions.keys())[0]))
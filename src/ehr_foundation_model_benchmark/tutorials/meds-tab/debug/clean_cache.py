# get all .*.npz_cache in subdirectories of the current directory

import glob
import os
import shutil

# path = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-fix2-large/Ischemic_Stroke_final/tabularize'
# path = '/data/processed_datasets/processed_datasets/ehr_foundation_data/outputs/models/meds_tab/output/in_hospital_mortality_final/tabularize'
for task in ['Celiac', 'CLL', 'HTN', 'Ischemic_Stroke', 'MASLD', 'Osteoporosis', 'Pancreatic_Cancer', 'Schizophrenia',  'SLE', 'T2DM']: 
    path = f'/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-2year/{task}_final/tabularize'
    files = glob.glob(os.path.join(path, '**', '.*.npz_cache'), recursive=True)
    k = 0
    for file in files:
        try:
            # remove only if name.npz does not exist when .name.npz_cache exists
            base_name = os.path.splitext(file)[0].replace('/.', '/') 
            if os.path.exists(base_name + '.npz'):
                continue
            k += 1
            shutil.rmtree(file)  # Remove the directory containing the cache file
            print(f"Removed cache file: {file}")
        except Exception as e:
            print(f"Error removing {file}: {e}")
    print(f"Total cache files removed: {k}")
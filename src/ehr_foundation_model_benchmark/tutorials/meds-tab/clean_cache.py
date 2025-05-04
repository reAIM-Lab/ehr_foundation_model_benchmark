# get all .*.npz_cache in subdirectories of the current directory

import glob
import os
import shutil

path = '/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-fix2-large/readmission_final/tabularize'
files = glob.glob(os.path.join(path, '**', '.*.npz_cache'), recursive=True)
# print(files)
k = 0
for file in files:
    try:
        # remove only if name.npz does not exist when .name.npz_cache exists
        base_name = os.path.splitext(file)[0].replace('/.', '/')  # Get the base name without extension
        # print(base_name)
        if os.path.exists(base_name + '.npz'):
            # print(f"Skipping {file} as corresponding .npz file exists.")
            continue
        k += 1
        # exit()
        # os.remove(file)
        # remove directory
        # os.rmdir(file)  # Remove the directory containing the cache file
        # remove file (which is a nonempty directory)
        shutil.rmtree(file)  # Remove the directory containing the cache file
        print(f"Removed cache file: {file}")
    except Exception as e:
        print(f"Error removing {file}: {e}")

print(f"Total cache files removed: {k}")
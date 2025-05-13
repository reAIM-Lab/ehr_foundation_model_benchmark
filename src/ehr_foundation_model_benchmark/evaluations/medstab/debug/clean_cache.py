"""
Script to clean lock files after meds-tab-timeseries crashed.
"""

import glob
import os
import shutil

path = 'XX4/tabularize'

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
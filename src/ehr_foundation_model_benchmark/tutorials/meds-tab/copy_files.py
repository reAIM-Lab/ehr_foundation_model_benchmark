import glob as glob
from tqdm import tqdm
import shutil
import os

dry_run = False

path_in = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-fix2-large"
path_out = "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-2year"

files = glob.glob(path_in + '/*_final/tabularize/*/*/*/*/*.npz')
files = sorted(files)
files = [f for f in files if not "full" in f]
# print(files)
print(len(files))

files_copied = 0
for f in tqdm(files):
    file_out = f.replace(path_in, path_out)
    if not os.path.exists(file_out):
        # print(f, file_out)
        os.makedirs(os.path.dirname(file_out), exist_ok=True)
        shutil.copy(f, file_out)
        files_copied += 1
    # break

print(files_copied)
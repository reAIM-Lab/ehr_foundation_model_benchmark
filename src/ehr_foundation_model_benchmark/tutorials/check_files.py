folder = '/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ohdsi_cumc_deid_2023q4r3_v3_mapped/models/meds_tab/output-2year/AMI_final/tabularize/train/'

import glob
import os

ft = ['count', 'sum', 'sum_sqd', 'min', 'max']
done = []
chunks = {}

for file in glob.glob(folder + '*/730d/*/*.npz'):
    chunk = file.split('/')[-4]
    if chunk not in chunks:
        chunks[chunk] = []
    print(file)
    if "code" in file:
        continue
    chunks[chunk].append(os.path.basename(file))

print(chunks)
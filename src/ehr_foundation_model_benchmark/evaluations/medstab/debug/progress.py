import os
import time
from tqdm import tqdm

folder_src_path = "XX1/data"
folder_path = "XX4/tabularize"
n_agg = 6
n_windows = 6


def count_parquet_files(path):
    count = 0
    for root, _, files in os.walk(path):
        count += sum(1 for f in files if f.endswith('.parquet'))
    return count

shard_files = count_parquet_files(folder_src_path)
print(f"Found {shard_files} shard files in {folder_src_path}")

# CONFIGURE THESE
total_files_expected = n_agg*n_windows*shard_files           # <-- Set your total target here
check_interval = 60                   # seconds

def count_npz_files(path):
    count = 0
    for root, _, files in os.walk(path):
        count += sum(1 for f in files if f.endswith('.npz'))
    return count

def main():
    last_count = count_npz_files(folder_path)
    pbar = tqdm(total=total_files_expected, initial=last_count, desc="Files processed", unit="file", smoothing=0)
    print("Starting to monitor folder:", folder_path)
    print("Expected total files:", total_files_expected)
    print("Current count of .npz files:", last_count)
    print("Checking every", check_interval, "seconds...")
    # print("Current count of .npz files:", count_npz_files(folder_path))

    while last_count < total_files_expected:
        current_count = count_npz_files(folder_path)
        if current_count != last_count:
            pbar.update(current_count - last_count)
            last_count = current_count
        time.sleep(check_interval)

    pbar.close()
    print("Target number of files reached!")

if __name__ == "__main__":
    main()
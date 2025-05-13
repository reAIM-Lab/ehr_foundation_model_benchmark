"""
Script to be called by auto-sweep.py
It extracts meds-tab features for the task and create two files: 
`indices.parquet` to indicate which row corresponds to which label (subject_id, time) and
`features_combined.npz` that contains all the features concatenated horizontally (all windows/aggs) and vertically (unsharded)
The goal is to use it for the LR afterwards
"""

import sys
import os
import glob
import time
from tqdm import tqdm
import shutil

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import coo_array
from pathlib import Path

task = sys.argv[1]
base_path = sys.argv[2]
label_path = sys.argv[3]

features_path = f"{base_path}/{task}_final/tabularize"
codes_path = f"{base_path}/{task}_final/metadata"
labels_path = f"{label_path}/{task}"
output_path = f"{base_path}/{task}_final/tabularize_export"

def get_min_dtype(array: np.ndarray) -> np.dtype:
    """Get the minimal dtype that can represent the array.

    Args:
        array: The array to determine the minimal dtype for.

    Returns:
        The minimal dtype that can represent the array, or the array's dtype if it is non-numeric.

    Examples:
        >>> get_min_dtype(np.array([1, 2, 3])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e9)])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, int(1e18)])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, -128])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1.0, 2.0, 3.0])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, np.nan])) # doctest:+ELLIPSIS
        dtype('...')
        >>> get_min_dtype(np.array([1, 2, 3, "a"])) # doctest:+ELLIPSIS
        dtype('...')
    """
    if np.issubdtype(array.dtype, np.integer):
        return np.result_type(np.min_scalar_type(array.min()), array.max())
    elif np.issubdtype(array.dtype, np.floating):
        return np.result_type(np.float32)
        # For more precision, we could do this
        # try:
        #    array.astype(np.float32, copy=False)
        #    return np.float32
        # except OverflowError:
        #    return np.float64

    return array.dtype

def sparse_matrix_to_array(coo_matrix: coo_array) -> tuple[np.ndarray, tuple[int, int]]:
    """Converts a sparse matrix to a numpy array format with shape information.

    Args:
        coo_matrix: The sparse matrix to convert.

    Returns:
        A tuple of a numpy array ([data, row, col]) and the shape of the original matrix.
    """
    data, row, col = coo_matrix.data, coo_matrix.row, coo_matrix.col
    # Remove invalid indices
    valid_indices = (data == 0) | np.isnan(data)
    data = data[~valid_indices]
    row = row[~valid_indices]
    col = col[~valid_indices]
    # reduce dtypes
    if len(data):
        data = data.astype(get_min_dtype(data), copy=False)
        row = row.astype(get_min_dtype(row), copy=False)
        col = col.astype(get_min_dtype(col), copy=False)

    return np.array([data, row, col]), coo_matrix.shape


def store_matrix(coo_matrix: coo_array, fp_path: Path, do_compress: bool) -> None:
    """Stores a sparse matrix to disk as a .npz file.

    Args:
        coo_matrix: The sparse matrix to store.
        fp_path: The file path where the matrix will be stored.
    """
    array, shape = sparse_matrix_to_array(coo_matrix)
    if do_compress:
        np.savez_compressed(fp_path, array=array, shape=shape)
    else:
        np.savez(fp_path, array=array, shape=shape)


def load_tab(path):
    """Loads a sparse matrix from disk.

    Args:
        path: Path to the sparse matrix.

    Returns:
        The sparse matrix.

    Raises:
        ValueError: If the loaded array does not have exactly 3 rows, indicating an unexpected format.
    """
    npzfile = np.load(path)
    array, shape = npzfile["array"], npzfile["shape"]
    if array.shape[0] != 3:
        raise ValueError(f"Expected array to have 3 rows, but got {array.shape[0]} rows")
    data, row, col = array
    return sp.csc_matrix((data, (row, col)), shape=shape)

files = sorted(glob.glob(os.path.join(labels_path, "**/*.parquet")))
features_all = []
indices_subject_ids = []
indices_timestamps = []
bvs = []
n_rows = 0
k = 0
n_labels_per_split = {}
for file in (pbar:=tqdm(files, desc="Processing files", total=len(files))):
    filename = os.path.basename(file)
    task_name = filename.replace(".parquet", "")
    split = os.path.basename(os.path.dirname(file))
    # print(split, task_name)


    pbar.set_description(f"Processing {task_name} in split {split} ({n_rows} rows so far)")

    # Load the labels
    labels_df = pd.read_parquet(file)
    if labels_df.empty:
        print(f"No labels found for {task_name} in split {split}. Skipping.")
        continue

    sids = labels_df["subject_id"].values.tolist()
    timestamps = labels_df["prediction_time"].values.tolist()
    bv = labels_df["boolean_value"].values.tolist()

    indices_subject_ids.extend(sids)
    if split not in n_labels_per_split:
        n_labels_per_split[split] = len(sids)
    else:
        n_labels_per_split[split] += len(sids)
    indices_timestamps.extend(timestamps)
    bvs.extend(bv)

    feature_local_path = os.path.join(features_path, split, task_name)
    all_features_npz_files = sorted(glob.glob(os.path.join(feature_local_path, "**/**/*.npz")))
    tabs = []
    bad = False
    for npz_file in (pbar2:=tqdm(all_features_npz_files, desc=f"Processing {task_name} features", total=len(all_features_npz_files), leave=False)):
        npz_agg = os.path.basename(npz_file)
        npz_agg_type = os.path.basename((os.path.dirname(npz_file)))
        npz_window = os.path.basename(os.path.dirname(os.path.dirname(npz_file)))
        # print(f"Processing {npz_filename} for task {task_name} in split {split}")
        pbar2.set_description(f"Processing {npz_window}-{npz_agg_type}-{npz_agg}")
        # time.sleep(0.05)  # Simulate some processing time
        tab = load_tab(npz_file)
        if tab.shape[0] != len(sids):
            os.remove(npz_file)
            print(f"Removing {npz_file} for task {task_name} in split {split}")
            k += 1
            print(f"Warning: Number of features ({tab.shape[0]}) does not match number of labels ({len(sids)}) for {task_name} in split {split}.")
            # exit()
            bad = True
        tabs.append(tab)
        # print(tab.shape)

    # hstack all tabs
    # if len(tabs) == 0:
    #     print(f"No features found for {task_name} in split {split}. Skipping.")
    #     continue
       

    if bad:
        print(f"Warning: Number of features does not match number of labels for {task_name} in split {split}. Skipping.")
        continue
    features = sp.hstack(tabs, format="csc")

    n_rows += features.shape[0]
    features_all.append(features)
    # print(features.shape)
    # Load the labels
    # labels_df = pd.read
    # exit()

print(f"Total number of labels per split: {n_labels_per_split}")

print(k)
# exit()


os.makedirs(output_path, exist_ok=True)
print(f"Total number of rows processed: {n_rows}")
print(f"Total number of features: {len(features_all)}")
print(f'Total number of indices: {len(indices_subject_ids)}')
# Convert indices to parquet
indices_df = pd.DataFrame({
    "subject_id": indices_subject_ids,
    "prediction_time": indices_timestamps,
    'boolean_value': bvs
})
indices_df.to_parquet(os.path.join(output_path, "indices.parquet"), index=False)
print(f"Indices saved to {os.path.join(output_path, 'indices.parquet')}")
# exit()
print("Combining all features...")

features_all = sp.vstack(features_all, format="csc")
print(f"Total features shape: {features_all.shape}")

# Save the combined features to a file
output_file = os.path.join(output_path, "features_combined.npz")
print(f"Saving combined features to {output_file}")

fa_coo = features_all.tocoo()
store_matrix(fa_coo, output_file, do_compress=True)

print("Done")



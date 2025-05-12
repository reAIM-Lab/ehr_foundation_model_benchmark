"""
Code to rebuild a medstab compatible feature folder with a subsampled dataset

It is part of the pipeline:
# 1 - linear_probing-xgboost to generate labels
# 2 - reshard.py to restructure the labels
# 3 - this script to extract the features
# 4 - meds-tab-xgboost to train the model
"""

from pathlib import Path
from scipy.sparse import coo_array
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import os
import shutil

import sys
k = int(sys.argv[1])
task = sys.argv[2]
base_path = sys.argv[3]
input_dir = sys.argv[4]
original_label_dir = sys.argv[5]


input_label_dir=f"{base_path}/labels_{k}"
output_dir=f"{base_path}/output_{k}"

print(input_dir)
print(original_label_dir)

print(input_label_dir)
print(output_dir)


# TODO import from medstab
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


dfs_ol = {}
dfs_l = {}

import glob
import pandas as pd
rows = 0
files = list(sorted(glob.glob(f"{input_dir}/**/*.npz", recursive=True)))
for file in (pbar:=tqdm(files)):
    file_suffix = file.replace(input_dir, "")
    file_split = file_suffix.split("/")[1]
    file_shard = file_suffix.split("/")[2]
    output_file = file.replace(input_dir, output_dir)
    original_label_path = f"{original_label_dir}/{file_split}/{file_shard}.parquet"
    label_path = f"{input_label_dir}/{file_split}/{file_shard}.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    pbar.set_description(f"Processing {file_suffix}")

    if file_split == "held_out":
        # copy original label file to label path
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        shutil.copy(original_label_path, label_path)
        shutil.copy(file, output_file)
        continue

    if (file_split, file_shard) not in dfs_ol:
        ol_df = pd.read_parquet(original_label_path)
        dfs_ol[(file_split, file_shard)] = ol_df.reset_index()\
            .rename(columns={'index': 'row_id'}).set_index(["subject_id", "prediction_time"])
        # print(ol_df)

    if (file_split, file_shard) not in dfs_l:
        label_df = pd.read_parquet(label_path)
        dfs_l[(file_split, file_shard)] = label_df.set_index(["subject_id", "prediction_time"])

    df_l = dfs_l[(file_split, file_shard)]
    df_ol = dfs_ol[(file_split, file_shard)]

    dfg = df_l.join(
        df_ol,
        on=["subject_id", "prediction_time"],
        rsuffix="_l",
        how="inner",
    )
    # print(dfg)

    npz_data = load_tab(file)
    data_npz = npz_data[dfg.row_id.values]

    rows += data_npz.shape[0]

    data_npz_coo = data_npz.tocoo()
    store_matrix(data_npz_coo, output_file, do_compress=False)
    
print(f"Total rows: {rows}")
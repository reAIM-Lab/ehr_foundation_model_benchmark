from pathlib import Path
import scipy.sparse as sp
import numpy as np
import pandas as pd

k = 100
k = 10
suffix = ""
suffix = "-test"

path_gen_labels = f"/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/meds-tab_check/models_mic1_general_tiny_0.1_{k}/long_los_output{suffix}/long_los/labels"
path_gen_tab = f"/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/meds-tab_check/models_mic1_general_tiny_0.1_{k}/long_los_output{suffix}/long_los/task_cache"
path_gen_mt = f"/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/meds-tab_check/models_mic1_general_tiny_0.1_{k}/long_los_output{suffix}/metadata"

path_task_labels = f"/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/ohdsi_cumc_deid_2023q4r3_10000_sample_meds_unit_concatenated/task_labels/long_los_sharded"
path_task_tab = f"/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/meds-tab_check/models_mic1_task_tiny_0.1_{k}/long_los_output{suffix}/tabularize"
path_task_mt = f"/data/processed_datasets/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/ml4h_demo/meds-tab_check/models_mic1_task_tiny_0.1_{k}/long_los_output{suffix}/metadata"

matches = [
    (path_gen_labels, path_task_labels, "parquet"),
    (path_gen_tab, path_task_tab, "npz")
]

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

def checks():
    for gen_path, task_path, ext in matches:

        gen_files = set(sorted(set([str(f).replace(gen_path, "") for f in Path(gen_path).glob("**/*." + ext)])))
        task_files = set(sorted(set([str(f).replace(task_path, "") for f in Path(task_path).glob("**/*." + ext)])))

        print(gen_files)
        print(task_files)

        print(f"Comparing {gen_path} with {task_path}")
        print(f"Generated files: {len(gen_files)}")
        print(f"Task files: {len(task_files)}")

        missing_in_gen = task_files - gen_files
        missing_in_task = gen_files - task_files

        if missing_in_gen:
            print(f"Files missing in generated path: {missing_in_gen}")
        else:
            print("All task files found in generated path.")

        if missing_in_task:
            print(f"Files missing in task path: {missing_in_task}")
        else:
            print("All generated files found in task path.")

        # Compare the contents of the files
        if ext == "parquet":
            import pandas as pd

            for file in gen_files.intersection(task_files):
                gen_df = pd.read_parquet(gen_path + file)
                task_df = pd.read_parquet(task_path + file)
                task_df['label'] = task_df['boolean_value']
                task_df['time'] = task_df['prediction_time']
    
                # print(gen_df)
                # print(task_df)
                # exit()


                cols = ['subject_id', 'label', 'time']
                gen_df = gen_df[cols]
                task_df = task_df[cols]


                diff = gen_df.compare(task_df)

                if len(diff) > 0:
                    print(f"Data mismatch in file: {file}")

                    print(gen_df)
                    print(task_df)

                    # find the first row that differs
                    print(f"Differences found in file {file}:\n{diff}")
                    print(len(diff))

                    exit()
                else:
                    print(f"Data match in file: {file}")
        elif ext == "npz":
            # import numpy as np

            for file in gen_files.intersection(task_files):
                gen_data = load_tab(gen_path + file)
                task_data = load_tab(task_path + file)

                print(gen_data)
                print(task_data)

                equal = (gen_data != task_data).nnz == 0
                if not equal:
                    print(f"Data mismatch in file: {file}")
                    print(f"Generated data shape: {gen_data.shape}")
                    print(f"Task data shape: {task_data.shape}")

                    # Print the differences
                    diff = gen_data != task_data
                    print(f"Number of differing elements: {diff.nnz}, Number of not null elements in gen_data: {gen_data.nnz}, task_data: {task_data.nnz}")
                    # print Diff elements
                    diff_indices = diff.nonzero()
                    for i in range(diff.nnz):
                        row, col = diff_indices[0][i], diff_indices[1][i]
                        print(f"Difference at row {row}, col {col}: gen={gen_data[row, col]}, task={task_data[row, col]}")
                        break

                    # Optionally, you can print the actual data if needed
                    # print("Generated data:", gen_data.toarray())
                    # print("Task data:", task_data.toarray())
                else:
                    print(f"Data match in file: {file}")
                # Assuming the npz files contain arrays with the same keys
                # for key in gen_data.keys():
                #     if key in task_data:
                #         print(f"Comparing key: {key} in file: {file}")
                #         print(gen_data[key].shape, task_data[key].shape)
                #         if not np.array_equal(gen_data[key], task_data[key]):
                #             print(f"Data mismatch in file: {file}, key: {key}")
                #             print(f"Generated data: {gen_data[key]}")
                #             print(f"Task data: {task_data[key]}")
                #             # exit()
                #         else:
                #             print(f"Data match for key: {key} in file: {file}")
                #     else:
                #         print(f"Key {key} missing in task data for file: {file}")

        print("-" * 40)

    exit()

    # then match the files between labels and tabularized data
    # and check if same number of rows labels and col data
    infos = [
        (path_gen_labels, path_gen_tab, "parquet", "npz"),
        (path_task_labels, path_task_tab, "parquet", "npz")
    ]
    for label_path, tab_path, label_ext, tab_ext in infos:
        print(f"Checking labels in {label_path} and tabular data in {tab_path}")
        label_files = set(sorted(set([str(f).replace(label_path, "") for f in Path(label_path).glob("**/*." + label_ext)])))
        # tab_files = set(sorted(set([str(f).replace(tab_path, "") for f in Path(tab_path).glob("**/*." + tab_ext)])))

        # print(label_files, tab_files)

        # just take labels and append 'full/code/count.npz' instead of 'parquet'
        # tab_files = set([f.replace("parquet", "full/code/count.npz") for f in label_files])
        
        # now iterate through the files and check if they match
        for label_file in label_files:
            tab_file = label_file.replace(".parquet", "/full/code/count.npz")
            import pandas as pd
            # import numpy as np
            label_df = pd.read_parquet(label_path + label_file)
            tab_data = load_tab(tab_path + tab_file)

            # check if the number of rows match
            # print(label_df.shape)
            # print(tab_data.shape)
            if label_df.shape[0] != tab_data.shape[0]:
                print(f"Mismatch in number of rows for file: {label_file}")
                print(f"Label rows: {label_df.shape[0]}, Tabular data rows: {tab_data.shape[0]}")
            else:
                # print(f"Number of rows match for file: {label_file}")
                pass

checks()

# check same code and not just permutations
codes_gen = pd.read_parquet(path_gen_mt + "/codes.parquet")
codes_task = pd.read_parquet(path_task_mt + "/codes.parquet")

codes_gen.sort_values(by='code', inplace=True)
codes_task.sort_values(by='code', inplace=True)

print(codes_gen)
print(codes_task)

print(codes_gen[codes_gen['count'] >= 1])
print(codes_gen.nunique())
# def filter_to_codes(
#     code_metadata_fp: Path,
#     allowed_codes: list[str] | None,
#     min_code_inclusion_count: int | None,
#     min_code_inclusion_frequency: float | None,
#     max_include_codes: int | None,
# ) -> ListConfig[str]:
#     """Filters and returns codes based on allowed list and minimum frequency.

#     Args:
#         code_metadata_fp: Path to the metadata file containing code information.
#         allowed_codes: List of allowed codes, None means all codes are allowed.
#         min_code_inclusion_count: Minimum count a code must have to be included.
#         min_code_inclusion_frequency: The minimum frequency a code must have,
#             normalized by dividing its count by the total number of observations
#             across all codes in the dataset, to be included.
#         max_include_codes: Maximum number of codes to include (selecting the most
#             prevelent codes).

#     Returns:
#         Sorted list of the intersection of allowed codes (if they are specified) and filters based on
#         inclusion frequency.

#     Examples:
#         >>> from tempfile import NamedTemporaryFile
#         >>> with NamedTemporaryFile() as f:
#         ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
#         ...     filter_to_codes( f.name, ["A", "D"], 3, None, None)
#         ['D']
#         >>> with NamedTemporaryFile() as f:
#         ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
#         ...     filter_to_codes( f.name, None, None, .35, None)
#         ['E']
#         >>> with NamedTemporaryFile() as f:
#         ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
#         ...     filter_to_codes( f.name, None, None, None, 1)
#         ['E']
#         >>> with NamedTemporaryFile() as f:
#         ...     pl.DataFrame({"code": ["E", "D", "A"], "count": [4, 3, 2]}).write_parquet(f.name)
#         ...     filter_to_codes( f.name, ["A", "D"], 10, None, None)
#         Traceback (most recent call last):
#         ...
#         ValueError: Code filtering criteria ...
#         ...
#     """
#     feature_freqs = pl.read_parquet(code_metadata_fp)

#     if allowed_codes is not None:
#         feature_freqs = feature_freqs.filter(pl.col("code").is_in(allowed_codes))

#     if min_code_inclusion_frequency is not None:
#         if min_code_inclusion_frequency < 0 or min_code_inclusion_frequency > 1:
#             raise ValueError("min_code_inclusion_frequency must be between 0 and 1.")
#         dataset_size = feature_freqs["count"].sum()
#         feature_freqs = feature_freqs.filter((pl.col("count") / dataset_size) >= min_code_inclusion_frequency)

#     if min_code_inclusion_count is not None:
#         feature_freqs = feature_freqs.filter(pl.col("count") >= min_code_inclusion_count)

#     if max_include_codes is not None:
#         feature_freqs = feature_freqs.sort("count", descending=True).head(max_include_codes)

#     if len(feature_freqs["code"]) == 0:
#         raise ValueError(
#             f"Code filtering criteria leaves only 0 codes. Note that {feature_freqs.shape[0]} "
#             "codes are read in, try modifying the following kwargs:"
#             f"\n- tabularization.allowed_codes: {allowed_codes}"
#             f"\n- tabularization.min_code_inclusion_count: {min_code_inclusion_count}"
#             f"\n- tabularization.min_code_inclusion_frequency: {min_code_inclusion_frequency}"
#             f"\n- tabularization.max_include_codes: {max_include_codes}"
#         )
#     return ListConfig(sorted(feature_freqs["code"].to_list()))

# print("Checking codes in generated metadata and task metadata")
# if not codes_gen.equals(codes_task):
#     print("Codes do not match between generated and task metadata.")
#     print("Generated codes:")
#     print(codes_gen)
#     print("Task codes:")
#     print(codes_task)
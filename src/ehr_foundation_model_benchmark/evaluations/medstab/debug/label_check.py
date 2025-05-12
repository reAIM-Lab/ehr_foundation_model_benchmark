"""
Code to compare the labels and the features resulting from all caching and task specific featurization
"""
from pathlib import Path
import scipy.sparse as sp
import numpy as np
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, coo_matrix

k = 100
k = 10
suffix = ""
suffix = "-test"

path_gen_labels = f"XX3/models_mic1_general_tiny_0.1_{k}/long_los_output{suffix}/long_los/labels"
path_gen_tab = f"XX5/models_mic1_general_tiny_0.1_{k}/long_los_output{suffix}/long_los/task_cache"
path_gen_mt = f"XX4/models_mic1_general_tiny_0.1_{k}/long_los_output{suffix}/metadata"

path_task_labels = f"XX3/long_los_sharded"
path_task_tab = f"XX4/models_mic1_task_tiny_0.1_{k}/long_los_output{suffix}/tabularize"
path_task_mt = f"XX4/models_mic1_task_tiny_0.1_{k}/long_los_output{suffix}/metadata"

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
                print(gen_data.shape)
                print(task_data.shape)

                try:
                    equal = (gen_data != task_data).nnz == 0
                except:
                    equal = (gen_data == task_data)
                    print((gen_data.nnz), (task_data.nnz))
                    exit()
                    # for i in range(task_data.nnz):
                    #     row, col = task_data.nonzero()[0][i], task_data.nonzero()[1][i]
                    #     print(f"Row {row}, Col {col}: gen={gen_data[row, col]}, task={task_data[row, col]}")
                    #     val_gen = gen_data.data[i]
                    #     val_task = task_data.data[i]
                    #     print(val_gen, val_task)
                    #     input()


                # resize task_data to match gen_data if necessary
                
                # if task_data.shape != gen_data.shape:
                #     print(f"Resizing task_data from {task_data.shape} to {gen_data.shape}")
                #     # add additional columns with zeros
                #     # task_data = sp.csc_matrix(
                #     #     (task_data.data, task_data.indices, task_data.indptr),
                #     #     shape=(task_data.shape[0], gen_data.shape[1])
                #     # )
                #     task_data, gen_data = align_sparse_matrices(task_data, gen_data)
                #     print(gen_data.shape, task_data.shape)
                #     print(gen_data)
                #     print(task_data)
                #     equal = (gen_data != task_data).nnz == 0
                #                     # Check if the data matches
                #     print(equal)
                #     exit()

                # print(gen_data.shape, task_data.shape)ßßß÷
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
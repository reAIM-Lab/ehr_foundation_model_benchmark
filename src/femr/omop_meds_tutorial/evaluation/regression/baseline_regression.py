import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import meds_reader
from tqdm import tqdm
from collections import defaultdict

cohort_dir = Path("/shared/share_mala/zj2398/mimic/regression/cohort/icu_stay_4h/")
for task in ["bilirubin"]:
    cohort_path =  cohort_dir/task/f"{task}.parquet"
    cohort_df = pd.read_parquet(cohort_path)
    # split_path = "/shared/share_mala/zj2398/mimic/mimic-3.1-meds/MEDS_cohort/metadata/subject_splits.parquet"
    # split = pd.read_parquet(split_path)
    # merged = cohort_df.merge(split,on="subject_id",how="inner")

    # held_out_df = merged[merged["split"] == "held_out"]
    # train_df = merged[merged["split"] != "held_out"]

    split_path = "/user/zj2398/cache/mtpp_8k/main_split.csv"
    split = pd.read_csv(split_path)
    split = split.rename(columns = {"split_name":"split"})
    merged = cohort_df.merge(split,on="subject_id",how="inner")
    test_df = merged[merged["split"] == "test"]
    train_df = merged[merged["split"] != "test"]

    print(train_df)
    print(test_df)

    meds_path = "/shared/share_mala/zj2398/mimic/meds_v0.6_reader"
    # Step 1: Calculate fallback values (mean per code in train_df)
    print("Calculating code means from training data...")
    code_means = train_df.groupby('code')['numerical_value'].mean().to_dict()
    print(f"Calculated means for {len(code_means)} unique codes")

    # Step 2: Load MEDS database
    print(f"Loading MEDS database from {meds_path}...")
    database = meds_reader.SubjectDatabase(meds_path)

    # Step 3: Organize test data by subject_id for efficient processing
    print("Organizing test data by subject...")
    test_by_subject = defaultdict(list)
    for idx, row in test_df.iterrows():
        test_by_subject[row['subject_id']].append({
            'index': idx,
            'prediction_time': row['prediction_time'],
            'code': row['code'],
            'true_value': row['numerical_value']
        })

    # Step 4: Define function to process each subject
    def process_subject(subject_id):
        """
        Process all test rows for a single subject.
        Returns list of (index, prediction) tuples.
        """
        if subject_id not in test_by_subject:
            return []
        
        try:
            subject = database[subject_id]
        except KeyError:
            # Subject not in database, use means for all predictions
            results = []
            for test_row in test_by_subject[subject_id]:
                pred = code_means.get(test_row['code'], np.nan)
                results.append((test_row['index'], pred))
            return results
        
        # Build a dictionary of historical values for each code
        # Key: code, Value: list of (time, value) tuples
        historical_values = defaultdict(list)
        
        for event in subject.events:
            # Store all events with their times and values
            if event.time is not None:
                # Check if event has numerical_value attribute
                if hasattr(event, 'numerical_value') and event.numerical_value is not None:
                    historical_values[event.code].append((event.time, event.numerical_value))
        
        # Process each test row for this subject
        results = []
        for test_row in test_by_subject[subject_id]:
            code = test_row['code']
            prediction_time = test_row['prediction_time']
            
            # Find the most recent value for this code before prediction_time
            last_value = None
            
            if code in historical_values:
                # Events are already sorted by time, so we iterate through them
                for event_time, event_value in historical_values[code]:
                    if event_time < prediction_time:
                        last_value = event_value
                    else:
                        # Since sorted, we can break early
                        break
            
            # Use last_value if found, otherwise use mean
            if last_value is not None:
                pred = last_value
            else:
                pred = code_means.get(code, np.nan)
            
            results.append((test_row['index'], pred))
        
        return results

    # Step 5: Process all subjects (with optional parallelization)
    print("Generating naive predictions...")
    all_predictions = {}

    # Get unique subject_ids from test data
    unique_subjects = list(test_by_subject.keys())

    for subject_id in tqdm(unique_subjects):
        subject_results = process_subject(subject_id)
        for idx, pred in subject_results:
            all_predictions[idx] = pred

    # Step 6: Add predictions to test_df
    test_df['prediction'] = test_df.index.map(all_predictions)

    # Step 7: Calculate evaluation metrics
    print("\nCalculating metrics...")
    valid_mask = ~test_df['prediction'].isna()
    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()

    print(f"Valid predictions: {n_valid} ({100*n_valid/len(test_df):.2f}%)")
    print(f"Invalid predictions: {n_invalid} ({100*n_invalid/len(test_df):.2f}%)")

    if n_valid > 0:
        rmse = float(np.sqrt(mean_squared_error(
            test_df.loc[valid_mask, "numerical_value"], 
            test_df.loc[valid_mask, 'prediction']
        )))
        mae = float(mean_absolute_error(
            test_df.loc[valid_mask, "numerical_value"], 
            test_df.loc[valid_mask, 'prediction']
        ))
        
        print(f"\n{'='*50}")
        print(f"Naive Baseline Results:")
        print(f"{'='*50}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"{'='*50}")
        
        # Optional: Show some statistics
        print(f"\nPrediction statistics:")
        print(f"Mean prediction: {test_df.loc[valid_mask, 'prediction'].mean():.4f}")
        print(f"Std prediction:  {test_df.loc[valid_mask, 'prediction'].std():.4f}")
        print(f"Mean true value: {test_df.loc[valid_mask, 'numerical_value'].mean():.4f}")
        print(f"Std true value:  {test_df.loc[valid_mask, 'numerical_value'].std():.4f}")
    else:
        print("No valid predictions to evaluate!")

# Optional: Save results
# test_df.to_csv("naive_baseline_results.csv", index=False)
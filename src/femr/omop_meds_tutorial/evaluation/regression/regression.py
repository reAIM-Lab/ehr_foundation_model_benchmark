import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import meds_reader
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

# ==================== CONFIGURATION ====================
COHORT_DIR = Path("/shared/share_mala/zj2398/mimic/regression/cohort/icu_stay_4h/")
SPLIT_PATH = "/user/zj2398/cache/mtpp_8k/main_split.csv"
MEDS_PATH = "/shared/share_mala/zj2398/mimic/meds_v0.6_reader"
OUTPUT_DIR = Path("./baseline_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Tasks to run
TASKS = ["creatinine", "platelets","pao2"]  # Add more tasks here: ["bilirubin", "creatinine", "glucose"]

# ==================== HELPER FUNCTIONS ====================

def load_cohort_data(cohort_dir, task, split_path):
    """Load and merge cohort data with split information."""
    print(f"\n{'='*60}")
    print(f"Loading cohort data for task: {task}")
    print(f"{'='*60}")
    
    cohort_path = cohort_dir / task / f"{task}.parquet"
    cohort_df = pd.read_parquet(cohort_path)
    print(f"Loaded {len(cohort_df)} samples from cohort")
    
    split = pd.read_csv(split_path)
    split = split.rename(columns={"split_name": "split"})
    
    merged = cohort_df.merge(split, on="subject_id", how="inner")
    print(f"After merging with splits: {len(merged)} samples")
    
    test_df = merged[merged["split"] == "test"].copy()
    train_df = merged[merged["split"] != "test"].copy()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Unique subjects in test: {test_df['subject_id'].nunique()}")
    
    return train_df, test_df


def calculate_code_means(train_df):
    """Calculate mean values for each code from training data."""
    code_means = train_df.groupby('code')['numerical_value'].mean().to_dict()
    print(f"Calculated means for {len(code_means)} unique codes")
    return code_means


def organize_test_data(test_df):
    """Organize test data by subject_id for efficient processing."""
    test_by_subject = defaultdict(list)
    
    # Convert prediction_time to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(test_df['prediction_time']):
        test_df['prediction_time'] = pd.to_datetime(test_df['prediction_time'])
    
    for idx, row in test_df.iterrows():
        test_by_subject[row['subject_id']].append({
            'index': idx,
            'prediction_time': pd.Timestamp(row['prediction_time']),
            'code': row['code'],
            'true_value': row['numerical_value']
        })
    
    print(f"Organized test data for {len(test_by_subject)} unique subjects")
    return test_by_subject


def process_subject(subject_id, database, test_by_subject, code_means):
    """
    Process all test rows for a single subject using naive forecasting.
    
    Returns:
        List of (index, prediction) tuples
    """
    if subject_id not in test_by_subject:
        return []
    
    # Get subject data from database
    try:
        subject = database[subject_id]
    except (KeyError, Exception) as e:
        # Subject not in database, use means for all predictions
        results = []
        for test_row in test_by_subject[subject_id]:
            pred = code_means.get(test_row['code'], np.nan)
            results.append((test_row['index'], pred))
        return results
    
    # Get all test rows for this subject (sorted by time for efficiency)
    test_rows = sorted(test_by_subject[subject_id], 
                       key=lambda x: x['prediction_time'])
    
    # Get unique codes we need to track
    needed_codes = set(row['code'] for row in test_rows)
    
    # Build historical values only for codes we need
    # Key: code, Value: list of (time, value) tuples
    historical_values = defaultdict(list)
    
    for event in subject.events:
        # Only process events for codes we care about
        if event.code not in needed_codes:
            continue
            
        if event.time is not None:
            # Check if event has numerical_value attribute
            if hasattr(event, 'numerical_value') and event.numerical_value is not None:
                event_time = pd.Timestamp(event.time)
                historical_values[event.code].append((event_time, event.numerical_value))
    
    # Process each test row for this subject
    results = []
    for test_row in test_rows:
        code = test_row['code']
        prediction_time = test_row['prediction_time']
        
        # Find the most recent value for this code before prediction_time
        last_value = None
        
        if code in historical_values:
            # Events are already sorted by time, iterate in reverse for efficiency
            for event_time, event_value in reversed(historical_values[code]):
                if event_time < prediction_time:
                    last_value = event_value
                    break
        
        # Use last_value if found, otherwise use mean
        pred = last_value if last_value is not None else code_means.get(code, np.nan)
        results.append((test_row['index'], pred))
    
    return results


def generate_predictions(database, test_by_subject, code_means, use_parallel=False):
    """Generate naive predictions for all test samples."""
    print("\nGenerating naive predictions...")
    all_predictions = {}
    unique_subjects = list(test_by_subject.keys())
    
    if use_parallel:
        # TODO: Implement parallel processing using database.map()
        # This would require restructuring the function to work with map
        print("Parallel processing not yet implemented, using sequential...")
    
    # Sequential processing with progress bar
    for subject_id in tqdm(unique_subjects, desc="Processing subjects"):
        subject_results = process_subject(subject_id, database, test_by_subject, code_means)
        for idx, pred in subject_results:
            all_predictions[idx] = pred
    
    return all_predictions


def evaluate_predictions(test_df, predictions_dict):
    """Calculate and display evaluation metrics."""
    test_df = test_df.copy()
    test_df['prediction'] = test_df.index.map(predictions_dict)
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    valid_mask = ~test_df['prediction'].isna()
    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()
    
    print(f"\nPrediction Coverage:")
    print(f"  Valid predictions: {n_valid} ({100*n_valid/len(test_df):.2f}%)")
    print(f"  Invalid predictions: {n_invalid} ({100*n_invalid/len(test_df):.2f}%)")
    
    metrics = {}
    if n_valid > 0:
        y_true = test_df.loc[valid_mask, "numerical_value"]
        y_pred = test_df.loc[valid_mask, 'prediction']
        
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        
        metrics['rmse'] = rmse
        metrics['mae'] = mae
        metrics['n_valid'] = int(n_valid)
        metrics['n_total'] = len(test_df)
        metrics['coverage'] = float(n_valid / len(test_df))
        
        print(f"\nPerformance Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        
        print(f"\nDistribution Statistics:")
        print(f"  Predictions - Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}")
        print(f"  True Values - Mean: {y_true.mean():.4f}, Std: {y_true.std():.4f}")
    else:
        print("\nERROR: No valid predictions to evaluate!")
    
    print("="*60)
    
    return test_df, metrics


def save_results(test_df, metrics, task, output_dir):
    """Save predictions and metrics to files."""
    # Save predictions
    pred_path = output_dir / f"{task}_naive_predictions.parquet"
    test_df.to_parquet(pred_path, index=False)
    print(f"\nSaved predictions to: {pred_path}")
    
    # Save metrics
    metrics_path = output_dir / f"{task}_naive_metrics.json"
    import json
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


# ==================== MAIN PIPELINE ====================

def run_naive_baseline(task, cohort_dir, split_path, meds_path, output_dir):
    """Run naive baseline for a single task."""
    
    # 1. Load data
    train_df, test_df = load_cohort_data(cohort_dir, task, split_path)
    
    # 2. Calculate code means from training data
    print("\nCalculating fallback values...")
    code_means = calculate_code_means(train_df)
    
    # 3. Load MEDS database
    print(f"\nLoading MEDS database from {meds_path}...")
    database = meds_reader.SubjectDatabase(meds_path)
    print(f"Database loaded successfully")
    
    # 4. Organize test data
    print("\nOrganizing test data...")
    test_by_subject = organize_test_data(test_df)
    
    # 5. Generate predictions
    predictions = generate_predictions(database, test_by_subject, code_means)
    
    # 6. Evaluate predictions
    test_df_with_preds, metrics = evaluate_predictions(test_df, predictions)
    
    # 7. Save results
    save_results(test_df_with_preds, metrics, task, output_dir)
    
    return metrics


# ==================== RUN FOR ALL TASKS ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NAIVE BASELINE - MULTIVARIATE TIME SERIES FORECASTING")
    print("="*60)
    
    all_metrics = {}
    
    for task in TASKS:
        try:
            metrics = run_naive_baseline(
                task=task,
                cohort_dir=COHORT_DIR,
                split_path=SPLIT_PATH,
                meds_path=MEDS_PATH,
                output_dir=OUTPUT_DIR
            )
            all_metrics[task] = metrics
        except Exception as e:
            print(f"\n ERROR processing task '{task}': {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY - ALL TASKS")
    print("="*60)
    for task, metrics in all_metrics.items():
        if metrics:
            print(f"\n{task.upper()}:")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  MAE:  {metrics.get('mae', 'N/A'):.4f}")
            print(f"  Coverage: {metrics.get('coverage', 'N/A'):.2%}")
    print("="*60)
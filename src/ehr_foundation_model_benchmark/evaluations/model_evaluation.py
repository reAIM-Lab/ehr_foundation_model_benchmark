import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, ci=0.95, seed=42):
    """
    Calculate confidence intervals for ROC-AUC and PR-AUC using bootstrap.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted probabilities for the positive class
    n_bootstraps : int, default=1000
        Number of bootstrap samples
    ci : float, default=0.95
        Confidence interval level
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary containing ROC-AUC and PR-AUC scores with their confidence intervals
    """
    # Set random seed
    np.random.seed(seed)

    # Calculate the original metrics
    original_roc_auc = roc_auc_score(y_true, y_pred)
    original_pr_auc = average_precision_score(y_true, y_pred)

    # Initialize arrays to store bootstrap results
    roc_aucs = np.zeros(n_bootstraps)
    pr_aucs = np.zeros(n_bootstraps)

    # Define indices for bootstrapping
    indices = np.arange(len(y_true))

    # Perform bootstrap sampling
    for i in range(n_bootstraps):
        # Sample with replacement
        bootstrap_indices = np.random.choice(indices, size=len(indices), replace=True)
        bootstrap_y_true = np.array(y_true)[bootstrap_indices]
        bootstrap_y_pred = np.array(y_pred)[bootstrap_indices]

        # Handle edge cases where only one class is present
        if len(np.unique(bootstrap_y_true)) < 2:
            # Skip this bootstrap if there's only one class
            continue

        # Calculate metrics
        roc_aucs[i] = roc_auc_score(bootstrap_y_true, bootstrap_y_pred)
        pr_aucs[i] = average_precision_score(bootstrap_y_true, bootstrap_y_pred)

    # Calculate confidence intervals
    alpha = (1 - ci) / 2
    ci_lower = alpha * 100
    ci_upper = (1 - alpha) * 100

    roc_auc_lower = np.percentile(roc_aucs, ci_lower)
    roc_auc_upper = np.percentile(roc_aucs, ci_upper)

    pr_auc_lower = np.percentile(pr_aucs, ci_lower)
    pr_auc_upper = np.percentile(pr_aucs, ci_upper)

    # Return results
    return {
        'roc_auc': {
            'score': original_roc_auc,
            'ci_lower': roc_auc_lower,
            'ci_upper': roc_auc_upper
        },
        'pr_auc': {
            'score': original_pr_auc,
            'ci_lower': pr_auc_lower,
            'ci_upper': pr_auc_upper
        }
    }


def evaluate_predictions(prediction_dir, evaluation_dir, n_bootstraps=1000, ci=0.95):
    """
    Evaluate prediction files and save metrics to the evaluation directory.
    Assumes all parquet files in the directory are for the same task.

    Parameters:
    -----------
    prediction_dir : str
        Directory containing parquet files with predictions
    evaluation_dir : str
        Directory to save evaluation results
    n_bootstraps : int, default=1000
        Number of bootstrap samples
    ci : float, default=0.95
        Confidence interval level
    """
    # Create evaluation directory if it doesn't exist
    os.makedirs(evaluation_dir, exist_ok=True)

    # Get task name from directory name
    task_name = os.path.basename(os.path.normpath(prediction_dir))

    print(f"Processing task: {task_name}")
    print(f"Loading parquet files from {prediction_dir}")

    # Load all parquet files in the directory
    try:
        # This loads all parquet files in the directory into a single DataFrame
        all_predictions = pd.read_parquet(prediction_dir)

        print(f"Loaded {len(all_predictions)} prediction rows")

        # Verify required columns exist
        if 'boolean_value' not in all_predictions.columns or 'predicted_boolean_probability' not in all_predictions.columns:
            print("Error: Data is missing required columns (boolean_value, predicted_boolean_probability)")
            return

        # Extract true labels and predictions
        y_true = all_predictions['boolean_value'].values
        y_pred = all_predictions['predicted_boolean_probability'].values

        # Check if there are at least two classes
        if len(np.unique(y_true)) < 2:
            print(f"Warning: Task {task_name} contains only one class, cannot calculate AUC metrics")
            return

        # Calculate class distribution
        positive_count = np.sum(y_true)
        total_count = len(y_true)
        positive_percentage = (positive_count / total_count) * 100

        print(f"Class distribution: {positive_count}/{total_count} positive examples ({positive_percentage:.2f}%)")

        # Calculate metrics with confidence intervals
        results = bootstrap_auc_ci(
            y_true=y_true,
            y_pred=y_pred,
            n_bootstraps=n_bootstraps,
            ci=ci
        )

        # Save results
        output_path = os.path.join(evaluation_dir, f"{task_name}_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        # Also save summary as CSV for easy reading
        summary_df = pd.DataFrame([{
            'task': task_name,
            'positive_examples': positive_count,
            'total_examples': total_count,
            'positive_percentage': positive_percentage,
            'roc_auc': results['roc_auc']['score'],
            'roc_auc_ci_lower': results['roc_auc']['ci_lower'],
            'roc_auc_ci_upper': results['roc_auc']['ci_upper'],
            'pr_auc': results['pr_auc']['score'],
            'pr_auc_ci_lower': results['pr_auc']['ci_lower'],
            'pr_auc_ci_upper': results['pr_auc']['ci_upper']
        }])

        summary_csv_path = os.path.join(evaluation_dir, f"{task_name}_metrics.csv")
        summary_df.to_csv(summary_csv_path, index=False)

        # Print summary
        print(f"\nResults for {task_name}:")
        print(
            f"ROC-AUC: {results['roc_auc']['score']:.3f} (95% CI: {results['roc_auc']['ci_lower']:.3f}-{results['roc_auc']['ci_upper']:.3f})")
        print(
            f"PR-AUC: {results['pr_auc']['score']:.3f} (95% CI: {results['pr_auc']['ci_lower']:.3f}-{results['pr_auc']['ci_upper']:.3f})")
        print(f"\nResults saved to {output_path} and {summary_csv_path}")

    except Exception as e:
        print(f"Error processing predictions: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Calculate ROC-AUC and PR-AUC metrics with confidence intervals")
    parser.add_argument("--prediction_dir", required=True, help="Directory containing prediction parquet files")
    parser.add_argument("--evaluation_dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--n_bootstraps", type=int, default=1000, help="Number of bootstrap samples")
    parser.add_argument("--confidence_interval", type=float, default=0.95, help="Confidence interval level")

    args = parser.parse_args()

    evaluate_predictions(
        prediction_dir=args.prediction_dir,
        evaluation_dir=args.evaluation_dir,
        n_bootstraps=args.n_bootstraps,
        ci=args.confidence_interval
    )


if __name__ == "__main__":
    main()
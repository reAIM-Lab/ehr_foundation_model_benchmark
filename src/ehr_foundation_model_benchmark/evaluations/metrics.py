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


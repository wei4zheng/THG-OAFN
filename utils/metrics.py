"""
Evaluation Metrics Module
Implements AUC, Recall, Precision, F1-score, and other metrics.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (used for AUC)

    Returns:
        metrics: Dictionary containing all computed metrics
    """
    metrics = {}

    # Check for NaN or Inf values
    if y_prob is not None:
        if np.isnan(y_prob).any() or np.isinf(y_prob).any():
            print("  Warning: y_prob contains NaN or Inf, setting AUC to 0.5")
            metrics['AUC'] = 0.5
        else:
            # Check if only one class is present
            unique_labels = np.unique(y_true)
            if len(unique_labels) < 2:
                print(f"  Warning: Only one class present in y_true: {unique_labels}, AUC not defined")
                metrics['AUC'] = 0.5
            else:
                try:
                    metrics['AUC'] = roc_auc_score(y_true, y_prob)
                except ValueError as e:
                    print(f"  Warning: AUC calculation failed: {e}, setting to 0.5")
                    metrics['AUC'] = 0.5
    else:
        metrics['AUC'] = 0.5

    # Compute Recall
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)

    # Compute Precision
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)

    # Compute F1-score
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)

    # Compute Accuracy
    metrics['Accuracy'] = (y_true == y_pred).sum() / len(y_true)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['Confusion_Matrix'] = cm

    return metrics


def print_metrics(metrics, dataset_name=""):
    """Print evaluation metrics"""
    print(f"\n{'='*50}")
    if dataset_name:
        print(f"Results on {dataset_name} dataset:")
    print(f"{'='*50}")

    for key, value in metrics.items():
        if key != 'Confusion_Matrix':
            if isinstance(value, float):
                print(f"{key:15s}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"{key:15s}: {value}")

    if 'Confusion_Matrix' in metrics:
        print(f"\nConfusion Matrix:")
        print(metrics['Confusion_Matrix'])
    print(f"{'='*50}\n")

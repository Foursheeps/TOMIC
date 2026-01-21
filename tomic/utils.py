"""
Utility functions for model evaluation and metrics computation.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Compute comprehensive metrics including accuracy, AUC, and F1 score.

    Args:
        preds (torch.Tensor): Predicted class logits of shape (batch_size, num_classes).
        labels (torch.Tensor): Ground truth labels of shape (batch_size,).

    Returns:
        dict: A dictionary containing:
            - accuracy: Overall accuracy
            - auc: AUC score (for binary classification)
            - auc_macro: AUC with macro averaging (for multi-class)
            - auc_weighted: AUC with weighted averaging (for multi-class)
            - f1_macro: F1 score with macro averaging
            - f1_micro: F1 score with micro averaging
            - f1_weighted: F1 score with weighted averaging
    """
    # Convert to numpy for sklearn metrics
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Get predicted classes
    predicted_classes = np.argmax(preds_np, axis=1)

    # Accuracy
    accuracy = accuracy_score(labels_np, predicted_classes)

    # Get number of classes
    num_classes = preds_np.shape[1]
    probs = torch.softmax(preds, dim=1).detach().cpu().numpy()

    metrics = {
        "accuracy": accuracy,
        "auc": 0.0,
        "auc_macro": 0.0,
        "auc_weighted": 0.0,
        "f1_macro": 0.0,
        "f1_micro": 0.0,
        "f1_weighted": 0.0,
    }

    # AUC scores
    if num_classes == 2:
        # Binary classification: use roc_auc_score
        try:
            # Check if both classes are present
            unique_labels = np.unique(labels_np)
            if len(unique_labels) == 2:
                auc_score = roc_auc_score(labels_np, probs[:, 1])
                metrics["auc"] = auc_score
                metrics["auc_macro"] = auc_score  # Same as auc for binary
                metrics["auc_weighted"] = auc_score  # Same as auc for binary
            else:
                # Only one class present, set AUC to 0
                metrics["auc"] = 0.0
                metrics["auc_macro"] = 0.0
                metrics["auc_weighted"] = 0.0
        except (ValueError, IndexError):
            # Handle any edge cases
            metrics["auc"] = 0.0
            metrics["auc_macro"] = 0.0
            metrics["auc_weighted"] = 0.0
    else:
        # Multi-class: use one-vs-rest approach
        try:
            # Check if all classes are present
            unique_labels = np.unique(labels_np)
            if len(unique_labels) >= 2:
                # Macro-averaged AUC (one-vs-rest)
                auc_macro = roc_auc_score(labels_np, probs, multi_class="ovr", average="macro")
                metrics["auc_macro"] = auc_macro

                # Weighted-averaged AUC (one-vs-rest)
                auc_weighted = roc_auc_score(labels_np, probs, multi_class="ovr", average="weighted")
                metrics["auc_weighted"] = auc_weighted
            else:
                # Only one class present
                metrics["auc_macro"] = 0.0
                metrics["auc_weighted"] = 0.0
        except (ValueError, IndexError):
            # Handle any edge cases
            metrics["auc_macro"] = 0.0
            metrics["auc_weighted"] = 0.0

    # F1 scores
    try:
        f1_macro = f1_score(labels_np, predicted_classes, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_macro
    except ValueError:
        metrics["f1_macro"] = 0.0

    try:
        f1_micro = f1_score(labels_np, predicted_classes, average="micro", zero_division=0)
        metrics["f1_micro"] = f1_micro
    except ValueError:
        metrics["f1_micro"] = 0.0

    try:
        f1_weighted = f1_score(labels_np, predicted_classes, average="weighted", zero_division=0)
        metrics["f1_weighted"] = f1_weighted
    except ValueError:
        metrics["f1_weighted"] = 0.0

    return metrics


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy metric.

    Args:
        preds (torch.Tensor): Predicted class logits.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy.
    """
    _, predicted_classes = torch.max(preds, 1)
    correct = (predicted_classes == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    return accuracy

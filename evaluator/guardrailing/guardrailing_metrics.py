import logging
from typing import Any, Dict, List

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

log = logging.getLogger(__name__)


def compute_classification_metrics(
    ground_truth_labels: List[str],
    predicted_labels: List[str],
    negative_label: str = "safe",
) -> Dict[str, float]:
    """
    Compute classification metrics.
    """
    if not ground_truth_labels or not predicted_labels:
        log.warning("Empty labels provided for metric computation")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    if len(ground_truth_labels) != len(predicted_labels):
        raise ValueError(
            f"Label length mismatch: {len(ground_truth_labels)} vs {len(predicted_labels)}"
        )

    accuracy = accuracy_score(ground_truth_labels, predicted_labels)

    pos_label = "unsafe" if negative_label == "safe" else "safe"

    precision = precision_score(
        ground_truth_labels, predicted_labels, pos_label=pos_label, average="binary"
    )

    recall = recall_score(
        ground_truth_labels, predicted_labels, pos_label=pos_label, average="binary"
    )

    f1 = f1_score(
        ground_truth_labels, predicted_labels, pos_label=pos_label, average="binary"
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def get_detailed_classification_report(
    ground_truth_labels: List[str],
    predicted_labels: List[str],
) -> Dict[str, Any]:
    """
    Get detailed classification report using sklearn.
    """
    if not ground_truth_labels or not predicted_labels:
        return {}

    return classification_report(
        ground_truth_labels, predicted_labels, output_dict=True
    )


def compute_all_metrics(
    ground_truth_labels: List[str],
    predicted_labels: List[str],
    negative_label: str = "safe",
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    """
    classification_metrics = compute_classification_metrics(
        ground_truth_labels, predicted_labels, negative_label
    )

    return {**classification_metrics}

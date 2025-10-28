"""
Classification Model Evaluation Metrics Toolbox

This module provides a unified interface for classification model evaluation metrics, supporting various metrics calculation and configuration.
Main features include:
- Standardized metric type definitions
- Flexible averaging method configuration
- Modern metric calculation based on torchmetrics
- Traditional Top-K accuracy calculation

Supported metric types:
- AUROC: Multiclass AUC curve
- MEAN_ACCURACY: Global average accuracy
- MEAN_RECALL: Global average recall
- MEAN_PER_CLASS_ACCURACY: Per-class average accuracy
- MEAN_PER_CLASS_RECALL: Per-class average recall
- PER_CLASS_ACCURACY: Per-class accuracy
- MEAN_MULTICLASS_F1: Global average F1 score
- MEAN_PER_CLASS_MULTICLASS_F1: Per-class average F1 score

Author: xuefeng zheng
Date: 2025-10-27
"""

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,      # Multiclass accuracy
    MulticlassAUROC,         # Multiclass AUC curve
    MulticlassF1Score,       # Multiclass F1 score
    MulticlassRecall,        # Multiclass recall
    MultilabelAveragePrecision,  # Multilabel average precision (unused)
    MultilabelF1Score,           # Multilabel F1 score (unused)
    MultilabelPrecisionRecallCurve,  # Multilabel PR curve (unused)
)
from torchmetrics.utilities.data import dim_zero_cat, select_topk


class AveragingMethod(Enum):
    """
    Averaging method enumeration for metrics

    Defines commonly used averaging methods in multiclass tasks:
    - micro: Global average, all samples equally weighted
    - macro: Class average, all classes equally weighted
    - none: No averaging, return individual results for each class
    """
    MEAN_ACCURACY = "micro"           # Global average accuracy
    MEAN_RECALL = "micro"             # Global average recall
    MEAN_PER_CLASS_ACCURACY = "macro" # Per-class average accuracy
    MEAN_PER_CLASS_RECALL = "macro"   # Per-class average recall
    PER_CLASS_ACCURACY = "none"       # Per-class accuracy (no averaging)
    MEAN_MULTICLASS_F1 = "micro"      # Global average F1 score
    MEAN_PER_CLASS_MULTICLASS_F1 = "macro"  # Per-class average F1 score

    def __str__(self) -> str:
        """Return the string representation of the enum value"""
        return self.value


class ClassificationMetricType(Enum):
    """
    Classification model evaluation metric type enumeration

    Defines all supported classification evaluation metric types, each enum value corresponds to one evaluation method.
    Automatically associates with the corresponding averaging method through the averaging_method property.
    """
    AUROC = "auroc"                          # Multiclass AUC curve
    MEAN_ACCURACY = "mean_accuracy"          # Global average accuracy (micro average)
    MEAN_RECALL = "mean_recall"              # Global average recall (micro average)
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"  # Per-class average accuracy (macro average)
    MEAN_PER_CLASS_RECALL = "mean_per_class_recall"      # Per-class average recall (macro average)
    PER_CLASS_ACCURACY = "per_class_accuracy"            # Per-class accuracy (no averaging)
    MEAN_MULTICLASS_F1 = "mean_multiclass_f1"            # Global average F1 score (micro average)
    MEAN_PER_CLASS_MULTICLASS_F1 = "mean_per_class_multiclass_f1"  # Per-class average F1 score (macro average)

    @property
    def averaging_method(self) -> Optional[AveragingMethod]:
        """
        Get the corresponding averaging method

        Returns:
            AveragingMethod: The corresponding averaging method enum value, or None if no correspondence
        """
        return getattr(AveragingMethod, self.name, None)

    def __str__(self) -> str:
        """Return the string representation of the enum value"""
        return self.value


def build_classification_metric(metric_type: ClassificationMetricType, num_classes: int) -> MetricCollection:
    """
    Build classification metric calculator

    Create the corresponding torchmetrics metric calculator based on the specified metric type and number of classes.
    Returns a MetricCollection wrapped object for unified interface calls.

    Args:
        metric_type (ClassificationMetricType): The metric type to create
        num_classes (int): The number of classes in the classification task

    Returns:
        MetricCollection: Collection containing the specified metric calculator

    Raises:
        ValueError: Raised when the metric type is not supported

    Examples:
        >>> f1_metric = build_classification_metric(ClassificationMetricType.MEAN_MULTICLASS_F1, 3)
        >>> acc_metric = build_classification_metric(ClassificationMetricType.MEAN_ACCURACY, 5)
    """
    # Get averaging method, raise exception if None
    avg_method = metric_type.averaging_method
    if avg_method is None:
        raise ValueError(f"No averaging method defined for metric type: {metric_type}")

    if metric_type in (
        ClassificationMetricType.MEAN_MULTICLASS_F1,
        ClassificationMetricType.MEAN_PER_CLASS_MULTICLASS_F1,
    ):
        # F1 score metric
        return MetricCollection({
            "f1": MulticlassF1Score(num_classes=num_classes, average=avg_method.value)
        })
    elif metric_type == ClassificationMetricType.MEAN_ACCURACY:
        # Accuracy metric
        return MetricCollection({
            "acc": MulticlassAccuracy(num_classes=num_classes, average=avg_method.value)
        })
    elif metric_type == ClassificationMetricType.MEAN_RECALL:
        # Recall metric
        return MetricCollection({
            "recall": MulticlassRecall(num_classes=num_classes, average=avg_method.value)
        })
    elif metric_type == ClassificationMetricType.AUROC:
        # AUC metric (AUROC usually does not need to specify averaging method)
        return MetricCollection({
            "auroc": MulticlassAUROC(num_classes=num_classes)
        })
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")


def accuracy(output: Tensor, target: Tensor, topk: tuple = (1,)) -> list:
    """
    Calculate Top-K accuracy (traditional implementation, not dependent on torchmetrics)

    Calculate whether the true label appears in the K categories with the highest prediction probability in the model prediction results.
    This is one of the most commonly used evaluation metrics in classification tasks.

    Args:
        output (Tensor): Model output logits, shape [batch_size, num_classes]
        target (Tensor): True labels, shape [batch_size]
        topk (tuple): List of K values to calculate, default only calculates top-1 accuracy

    Returns:
        list: List of accuracy percentages for each K value

    Examples:
        >>> # Calculate top-1 and top-5 accuracy
        >>> output = torch.randn(32, 1000)  # 32 samples, 1000 classes
        >>> target = torch.randint(0, 1000, (32,))  # True labels
        >>> top1_acc, top5_acc = accuracy(output, target, topk=(1, 5))
        >>> print(f"Top-1: {top1_acc:.2f}%, Top-5: {top5_acc:.2f}%")
    """
    maxk = max(topk)  # Get the largest K value
    batch_size = target.size(0)  # Batch size

    # Get the maxk categories with the highest prediction probability for each sample
    # output.topk(maxk, 1, True, True) returns (values, indices)
    # We only need indices (predicted category indices)
    _, pred = output.topk(maxk, 1, True, True)

    # Transpose prediction results: from [batch_size, maxk] to [maxk, batch_size]
    pred = pred.t()

    # Expand true labels for comparison with prediction results
    # target.view(1, -1) changes labels to [1, batch_size]
    # .expand_as(pred) expands to [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # Calculate accuracy for each K value
    # correct[:k] takes the first k prediction results
    # .reshape(-1) flattens to 1D
    # .float().sum(0) calculates the number of correct predictions
    # * 100.0 / batch_size converts to percentage
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]



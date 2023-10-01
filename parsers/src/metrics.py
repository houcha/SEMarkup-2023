from overrides import override
from typing import List, Optional

import torch
from torch import Tensor, BoolTensor
import torch.nn.functional as F

import numpy as np

from allennlp.training.metrics.metric import Metric


@Metric.register("accuracy")
class Accuracy(Metric):
    def __init__(self) -> None:
        self._correct_count = 0
        self._total_count = 0

    @override
    def __call__(self, test_labels: List, gold_labels: List, mask = None):
        if mask is None:
            mask = np.ones(len(test_labels))
        assert len(test_labels) == len(gold_labels) == len(mask)

        for test_label, gold_label, masked in zip(test_labels, gold_labels, mask):
            if not masked:
                continue
            if test_label == gold_label:
                self._correct_count += 1
            self._total_count += 1

    @override
    def get_metric(self, reset: bool = False) -> float:
        accuracy = self._correct_count / self._total_count if self._total_count > 0 else 0
        if reset:
            self.reset()
        return accuracy

    @override
    def reset(self):
        self._correct_count = 0
        self._total_count = 0


@Metric.register("cross-entropy")
class CrossEntropy(Metric):
    def __init__(self, weight: Optional[Tensor] = None) -> None:
        self._weight = weight
        self._cross_entropy = 0.0
        self._count = 0

    @override
    def __call__(
        self,
        predictions: Tensor,
        gold_labels: Tensor,
        mask: Optional[BoolTensor] = None
    ) -> Tensor:
        device = predictions.device

        if mask is None:
            mask = torch.ones(predictions.size()[:-1], device=device).bool()
        weight = self._weight.to(device) if self._weight is not None else None
        cross_entropy = F.cross_entropy(predictions[mask], gold_labels[mask], weight)

        self._cross_entropy += cross_entropy.item()
        self._count += 1
        return cross_entropy

    @override
    def get_metric(self, reset: bool = False) -> float:
        average_value = self._cross_entropy / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @override
    def reset(self) -> None:
        self._cross_entropy = 0.0
        self._count = 0


@Metric.register("identity-loss")
class IdentityLoss(Metric):
    """
    This loss practically just act as a buffer for loss.
    """
    def __init__(self) -> None:
        self._loss = 0.0
        self._count = 0

    @override
    def __call__(self, loss: Tensor) -> Tensor:
        self._loss += loss.item()
        self._count += 1
        return loss

    @override
    def get_metric(self, reset: bool = False) -> float:
        average_value = self._loss / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @override
    def reset(self) -> None:
        self._loss = 0.0
        self._count = 0


from overrides import override
from typing import List

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
        accuracy = 0
        if self._total_count > 0:
            accuracy = self._correct_count / self._total_count
        if reset:
            self.reset()
        return accuracy

    @override
    def reset(self):
        self._correct_count = 0
        self._total_count = 0


from overrides import override

from typing import Optional

import torch
import torch.nn.functional as F

from allennlp.training.metrics.metric import Metric


@Metric.register("cross-entropy")
class CrossEntropy(Metric):
    def __init__(self, weight: Optional[torch.Tensor] = None) -> None:
        self._weight = weight
        self._cross_entropy = 0.0
        self._count = 0

    @override
    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
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

from overrides import override
from typing import Dict, List, Optional

import torch
from torch import nn
from torch import Tensor

from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.training.metrics import CategoricalAccuracy, FBetaVerboseMeasure

from .vocabulary import VocabularyWeighted
from .cross_entropy import CrossEntropy


@Model.register('feed_forward_classifier')
class FeedForwardClassifier(Model):
    """
    A simple classifier composed of two feed-forward layers separated by a nonlinear activation.
    """
    def __init__(self,
        vocab: VocabularyWeighted,
        labels_namespace: str,
        in_dim: int,
        hid_dim: int,
        activation: str,
        dropout: float,
        use_class_weights: bool = False,
        head_loss_weight: float = 1.0
    ):
        super().__init__(vocab)

        n_classes = vocab.get_vocab_size(labels_namespace)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hid_dim),
            Activation.by_name(activation)(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, n_classes)
        )
        # Metrics.
        weight_vector = None
        if use_class_weights:
            weight_vector = vocab.get_weight_vector(labels_namespace)
        self.loss = CrossEntropy(weight=weight_vector)
        self.head_loss_weight = head_loss_weight
        self.accuracy = CategoricalAccuracy()
        self.fscore = FBetaVerboseMeasure()

    @override(check_signature=False)
    def forward(
        self,
        embeddings: Tensor,
        labels: Optional[Tensor] = None,
        mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:

        logits = self.classifier(embeddings)
        preds = logits.argmax(-1)

        loss = torch.tensor(0.)
        if labels is not None:
            loss = self.update_loss(logits, labels, mask)
            self.update_metrics(logits, labels, mask)

        return {'logits': logits, 'preds': preds, 'loss': loss}

    def update_loss(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return self.loss(logits, target, mask) * self.head_loss_weight

    def update_metrics(self, logits: Tensor, target: Tensor, mask: Tensor):
        self.accuracy(logits, target, mask)
        self.fscore(logits, target, mask)

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy.get_metric(reset),
            "macro-fscore": self.fscore.get_metric(reset)["macro-fscore"],
            "loss": self.loss.get_metric(reset)
        }


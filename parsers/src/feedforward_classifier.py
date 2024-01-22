from overrides import override
from typing import Dict, List, Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.training.metrics import CategoricalAccuracy, FBetaVerboseMeasure
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.modules.conditional_random_field.conditional_random_field import ConditionalRandomField

from .vocabulary import VocabularyWeighted
from .metrics import CrossEntropy, IdentityLoss


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
        head_loss_weight: float = 1.0,
        use_crf = False
    ):
        super().__init__(vocab)

        self.labels_namespace = labels_namespace
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

        if use_crf:
            self.crf = ConditionalRandomField(n_classes)
            self.loss = IdentityLoss()
        else:
            self.crf = None
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

        prediction_ids = logits.argmax(-1)
        if self.crf is not None and not self.training:
            paths_and_scores = self.crf.viterbi_tags(logits, mask)
            paths_tensors = [torch.Tensor(path) for path, score in paths_and_scores]
            padding_value = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN)
            prediction_ids = pad_sequence(paths_tensors, batch_first=True, padding_value=padding_value)

        loss = torch.tensor(0.0)
        if labels is not None:
            if self.crf is not None:
                batch_size = embeddings.shape[0]
                log_likelihood_avg = self.crf(logits, labels, mask) / batch_size
                loss = self.loss(-log_likelihood_avg) * self.head_loss_weight
            else:
                loss = self.loss(logits, labels, mask) * self.head_loss_weight
            self.update_metrics(logits, labels, mask)

        return {'logits': logits, 'prediction_ids': prediction_ids, 'loss': loss}

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

    def decode_ids(self, prediction_ids: Tensor, oov_token_replacement=None) -> List[List]:
        batch_size, sentence_max_length = prediction_ids.shape
        predictions = [[None for i in range(sentence_max_length)] for j in range(batch_size)]
        prediction_ids = prediction_ids.tolist()
        for i, sentence_predictions_ids in enumerate(prediction_ids):
            for j, prediction_id in enumerate(sentence_predictions_ids):
                predictions[i][j] = self.vocab.get_token_from_index(prediction_id, self.labels_namespace)
                if oov_token_replacement is not None and predictions[i][j] == DEFAULT_OOV_TOKEN:
                    predictions[i][j] = oov_token_replacement
        return predictions


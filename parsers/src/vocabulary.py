from typing import Dict

import torch

from allennlp.data.vocabulary import Vocabulary


def _calc_classes_weights(
    namespace_labels_counts: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, float]]:

    namespace_classes_weights = dict()
    for namespace, labels_counts in namespace_labels_counts.items():
        n_samples = sum(counts for counts in labels_counts.values())
        n_classes = len(labels_counts)

        classes_weights = dict()
        for label, counts in labels_counts.items():
            classes_weights[label] = n_samples / (n_classes * counts)

        namespace_classes_weights[namespace] = classes_weights
    return namespace_classes_weights


@Vocabulary.register("vocabulary_weighted", constructor="from_instances")
class VocabularyWeighted(Vocabulary):
    """
    An extension of allennlp built-in Vocabulary that can handle class weights.
    """

    def __init__(self, counter: Dict[str, Dict[str, int]] = None, **kwargs) -> None:
        super().__init__(counter, **kwargs)

        self.namespace_classes_weights = None
        if counter is not None:
            self.namespace_classes_weights = _calc_classes_weights(counter)

    def get_weight_vector(self, namespace: str) -> torch.Tensor:
        """
        Returns a vector of class weights for a given namespace.
        """
        class_weights = self.namespace_classes_weights[namespace]
        weight_vector = torch.zeros(len(class_weights))
        for label, weight in sorted(class_weights.items(), key=lambda x: x[1]):
            label_index = self.get_token_index(label, namespace)
            weight_vector[label_index] = weight
        return weight_vector


from typing import Dict

from allennlp.data.vocabulary import Vocabulary


def _calc_classes_weights(namespace_labels_counts: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, float]]:
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
    def __init__(self, counter: Dict[str, Dict[str, int]] = None, **kwargs) -> None:
        super().__init__(counter, **kwargs)

        self.namespace_classes_weights = _calc_classes_weights(counter)

    def get_class_weights(self, namespace: str) -> Dict[str, Dict[str, float]]:
        return self.namespace_classes_weights[namespace]


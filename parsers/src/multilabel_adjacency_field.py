from typing import List, Dict, Optional, Tuple
import textwrap

import torch

from allennlp.data.fields import Field, SequenceField
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError


class MultilabelAdjacencyField(Field[torch.Tensor]):
    def __init__(
        self,
        sequence_field: SequenceField,
        indices: List[Tuple[int, int]],
        labels: List[str],
        label_namespace: str = "labels",
        padding_value: int = -1,
    ) -> None:
        self.indices = indices
        self.labels = labels
        self.sequence_field = sequence_field
        self._num_labels: Optional[int] = None
        self._labels_ids: Optional[List[List[int]]] = None
        self._label_namespace = label_namespace
        self._padding_value = padding_value

        field_length = sequence_field.sequence_length()

        if len(set(indices)) != len(indices):
            raise ConfigurationError(f"Indices must be unique, but found {indices}")

        if not all(
            0 <= index[0] < field_length and 0 <= index[1] < field_length for index in indices
        ):
            raise ConfigurationError(
                f"Label indices and sequence length "
                f"are incompatible: {indices} and {field_length}"
            )

        if len(indices) != len(labels):
            raise ConfigurationError(
                f"Labels and indices lengths do not match: {labels}, {indices}"
            )

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._labels_ids is None:
            for label in self.labels:
                counter[self._label_namespace][label] += 1

    def index(self, vocab: Vocabulary):
        if self._labels_ids is None:
            self._labels_ids = [vocab.get_token_index(label, self._label_namespace) for label in self.labels]
        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": self.sequence_field.sequence_length()}

    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        seq_len = padding_lengths["num_tokens"]
        # Initialize all with padding value.
        matrix = torch.full((seq_len, seq_len), self._padding_value, dtype=torch.long)
        assert self._labels_ids is not None
        # Assign labels to edges.
        for index, label_id in zip(self.indices, self._labels_ids):
            matrix[index] = label_id
        return matrix

    def empty_field(self) -> "MultilabelAdjacencyField":

        # The empty_list here is needed for mypy
        empty_list: List[Tuple[int, int]] = []
        multilabel_adjacency_field = MultilabelAdjacencyField(
            empty_list, self.sequence_field.empty_field(), empty_list, padding_value=self._padding_value
        )
        return multilabel_adjacency_field

    def __str__(self) -> str:
        length = self.sequence_field.sequence_length()
        formatted_labels = "".join(
            "\t\t" + labels + "\n" for labels in textwrap.wrap(repr(self.labels), 100)
        )
        formatted_indices = "".join(
            "\t\t" + index + "\n" for index in textwrap.wrap(repr(self.indices), 100)
        )
        return (
            f"MultilabelAdjacencyField of length {length}\n"
            f"\t\twith indices:\n {formatted_indices}\n"
            f"\t\tand labels:\n {formatted_labels} \t\tin namespace: '{self._label_namespace}'."
        )

    def __len__(self):
        return len(self.sequence_field)

    def human_readable_repr(self):
        ret = {"indices": self.indices}
        if self.labels is not None:
            ret["labels"] = self.labels
        return ret


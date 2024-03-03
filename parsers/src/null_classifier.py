from typing import List, Dict

import torch
from torch import nn
from torch import Tensor, BoolTensor

from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.fields import TextField
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers import Token as AllenToken
from allennlp.data import TextFieldTensors
from allennlp.nn.util import get_text_field_mask, move_to_device, get_device_of, get_lengths_from_binary_sequence_mask
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, F1Measure

import sys
sys.path.append("..")
from common.token import Token, CLS_TOKEN

from .feedforward_classifier import FeedForwardClassifier


@Model.register('null_classifier')
class NullClassifier(FeedForwardClassifier):
    """
    Binary classifier of ellipted tokens.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        indexer: TokenIndexer,
        embedder: TokenEmbedder,
        in_dim: int,
        hid_dim: int,
        activation: str,
        dropout: float,
        positive_class_weight: float = 1.0
    ):
        super().__init__(
            vocab=vocab,
            in_dim=in_dim,
            hid_dim=hid_dim,
            n_classes=2,
            activation=activation,
            dropout=dropout
        )
        self.indexer = indexer
        self.embedder = embedder

        weight = torch.Tensor([1.0, positive_class_weight])
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.accuracy = CategoricalAccuracy()
        self.fscore = F1Measure(positive_label=1)

    def forward(
        self,
        words: TextFieldTensors,
        sentences: List[List[Token]],
        is_inference: bool
    ) -> Dict[str, Tensor]:

        device = get_device_of(words["tokens"]["token_ids"])

        words_with_nulls = words
        sentences_with_nulls = sentences
        # Extra [CLS] token accounts for the case when #NULL is the first token in a sentence.
        sentences_with_cls_and_nulls = self._add_cls_token(sentences_with_nulls)
        sentences_with_cls_and_no_nulls = self._remove_nulls(sentences_with_cls_and_nulls)

        target_has_null_after = self._build_has_null_after(sentences_with_cls_and_nulls).long()
        target_has_null_after = move_to_device(target_has_null_after, device)

        words_with_cls_and_no_nulls = self._create_words(sentences_with_cls_and_no_nulls, device)
        mask_no_nulls = get_text_field_mask(words_with_cls_and_no_nulls)
        embeddings_with_cls_and_no_nulls = self.embedder(**words_with_cls_and_no_nulls['tokens'])
        nulls = super().forward(embeddings_with_cls_and_no_nulls, target_has_null_after, mask_no_nulls)
        loss = nulls["loss"]

        if is_inference:
            sentences_with_cls_and_nulls = self._add_nulls(sentences_with_cls_and_no_nulls, nulls["preds"])
            sentences_with_nulls = self._remove_cls_token(sentences_with_cls_and_nulls)
            words_with_nulls = self._create_words(sentences_with_nulls, device=device)

        return words_with_nulls, sentences_with_nulls, loss

    def _create_words(self, sentences: List[List[Token]], device):
        text_fields = []
        max_padding_lengths = {}
        
        for sentence in sentences:
            #print(f"sentence: {sentence}")
            tokens = [AllenToken(token.form) for token in sentence]
            text_field = TextField(tokens, {"tokens": self.indexer})
            text_field.index(self.vocab)
            if not max_padding_lengths:
                max_padding_lengths = text_field.get_padding_lengths()
            for name, length in text_field.get_padding_lengths().items():
                max_padding_lengths[name] = max(max_padding_lengths[name], length)
            text_fields.append(text_field)
        
        tensors = []
        for text_field in text_fields:
            tensor = text_field.as_tensor(max_padding_lengths)
            tensors.append(move_to_device(tensor, device))
        assert 0 < len(text_fields)
        words = text_fields[0].batch_tensors(tensors)
        return words

    @staticmethod
    def _build_has_null_after(sentences: List[List[Token]]) -> BoolTensor:
        """
        Return mask without nulls, where mask[:, i] = True iff token i has null after him in original sentence (with nulls).
        """

        nulls = torch.nn.utils.rnn.pad_sequence(
            [torch.LongTensor([token.is_null() for token in sentence]) for sentence in sentences],
            batch_first=True,
            padding_value=-1
        )
        null_mask = (nulls == 1)
        no_null_mask = (nulls == 0)

        has_null_after = null_mask.roll(shifts=-1, dims=-1)
        sentences_without_nulls_lengths = get_lengths_from_binary_sequence_mask(no_null_mask).tolist()
        has_null_after_trimmed = torch.masked_select(has_null_after, no_null_mask).split(sentences_without_nulls_lengths)
        return torch.nn.utils.rnn.pad_sequence(has_null_after_trimmed, batch_first=True, padding_value=False)

    @staticmethod
    def _remove_nulls(sentences: List[List[Token]]):
        return [[token for token in sentence if not token.is_null()] for sentence in sentences]

    @staticmethod
    def _add_cls_token(sentences: List[List[Token]]):
        # Place token on the first position
        return [[CLS_TOKEN, *sentence] for sentence in sentences]

    @staticmethod
    def _remove_cls_token(sentences: List[List[Token]]):
        # Remove first token.
        return [sentence[1:] for sentence in sentences]

    @staticmethod
    def _add_nulls(sentences: List[List[Token]], nulls):
        sentences_with_nulls = []
        for sentence, nulls_sentence in zip(sentences, nulls):
            sentence_with_nulls = []
            for token, should_insert_null in zip(sentence, nulls_sentence):
                sentence_with_nulls.append(token)
                if should_insert_null:
                    sentence_with_nulls.append(Token.create_null(id=f"{token.id}.1"))
            sentences_with_nulls.append(sentence_with_nulls)
        return sentences_with_nulls

    def update_metrics(self, logits: Tensor, labels: Tensor, mask: Tensor):
        self.accuracy(logits, labels, mask)
        self.fscore(logits, labels, mask)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1measure = self.fscore.get_metric(reset)
        return {
            "Accuracy": self.accuracy.get_metric(reset),
            "f1": f1measure["f1"]
        }


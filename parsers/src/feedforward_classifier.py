from overrides import override
from typing import Dict, List, Optional

import re
import string
import logging

import torch
from torch import nn
from torch import Tensor

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.training.metrics import CategoricalAccuracy
from tqdm import tqdm

from .lemmatize_helper import LemmaRule, predict_lemma_from_rule, normalize, DEFAULT_LEMMA_RULE


logger = logging.getLogger(__name__)


@Model.register('feed_forward_classifier')
class FeedForwardClassifier(Model):
    """
    A simple classifier composed of two feed-forward layers separated by a nonlinear activation.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 in_dim: int,
                 hid_dim: int,
                 n_classes: int,
                 activation: str,
                 dropout: float):
        super().__init__(vocab)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hid_dim),
            Activation.by_name(activation)(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, n_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.metric = CategoricalAccuracy()

    @override(check_signature=False)
    def forward(self,
                embeddings: Tensor,
                labels: Tensor = None,
                mask: Tensor = None) -> Dict[str, Tensor]:
        logits = self.classifier(embeddings)
        preds = logits.argmax(-1)

        loss = torch.tensor(0.)
        if labels is not None:
            loss = self.loss(logits, labels, mask)
            self.metric(logits, labels, mask)

        return {'logits': logits, 'preds': preds, 'loss': loss}

    def loss(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        return self.criterion(logits[mask], target[mask])

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"Accuracy": self.metric.get_metric(reset)}


def is_punctuation(word: str) -> bool:
    return word in LemmaClassifier.PUNCTUATION

def contains_digit(word: str) -> bool:
    return any(char.isdigit() for char in word)

def is_title(word: str) -> bool:
    return word[0].isupper()


# TODO: move to separate file
@Model.register('lemma_classifier')
class LemmaClassifier(FeedForwardClassifier):
    """
    FeedForwardClassifier specialization for lemma classification.
    """

    PUNCTUATION = set(string.punctuation)
    COLLISION_TOKEN = "@@COLLISION@@"

    def __init__(self,
                 vocab: Vocabulary,
                 in_dim: int,
                 hid_dim: int,
                 n_classes: int,
                 activation: str,
                 dropout: float,
                 paradigm_dictionary_path: str = None,
                 dictionary_lemmas_info: List[Dict[str, str]] = [],
                 topk: int = 1):

        super().__init__(vocab, in_dim, hid_dim, n_classes, activation, dropout)

        self.paradigm_dictionary = dict()
        if paradigm_dictionary_path:
            logger.info("Reading paradigm dictionary...")
            self.paradigm_dictionary = self.build_paradigm_dictionary(paradigm_dictionary_path)
            logger.info(f"Dictionary size: {len(self.paradigm_dictionary)}")

        self.dictionary_lemmas = set()
        for lemmas_info in dictionary_lemmas_info:
            self.dictionary_lemmas |= self.find_lemmas(lemmas_info["path"], lemmas_info["lemma_match_pattern"])
        assert topk == 1 if not self.dictionary_lemmas else topk >= 1
        self.topk = topk

    def find_lemmas(self, file_path: str, lemma_match_pattern: str) -> set:
        with open(file_path, 'r') as f:
            txt = f.read()
            lemmas = re.findall(lemma_match_pattern, txt, re.MULTILINE)
            lemmas = set(map(normalize, lemmas))
        return lemmas

    def build_paradigm_dictionary(self, dictionary_path: str) -> dict:
        dictionary = dict()
        with open(dictionary_path, 'r') as f:
            while line := f.readline():
                # Skip newlines.
                if not line.strip():
                    continue

                lemma, _ = line.strip().split('\t')
                lemma_normalized = normalize(lemma)

                # Use set to avoid duplicates within a paradigm.
                unique_wordforms = set()
                unique_wordforms.add(lemma_normalized)
                while (line := f.readline()):
                    # Newline indicates the end of the paradigm group.
                    if not line.strip():
                        break
                    wordform, _ = line.strip().split('\t')
                    wordform_normalized = normalize(wordform)
                    unique_wordforms.add(wordform_normalized)

                for wordform in unique_wordforms:
                    dictionary[wordform] = lemma_normalized if wordform not in dictionary else LemmaClassifier.COLLISION_TOKEN
        return dictionary

    @override
    def forward(self,
                embeddings: Tensor,
                labels: Tensor = None,
                mask: Tensor = None,
                metadata: Dict = None) -> Dict[str, Tensor]:

        output = super().forward(embeddings, labels, mask)
        logits, _, loss = output['logits'].cpu(), output['preds'], output['loss']

        batch_size, sentence_max_length = logits.shape[0], logits.shape[1]
        # Find top most confident lemma rules for each token.
        top_rules = torch.argmax(logits, dim=-1).numpy()
        # Predict lemmas.
        predictions = batch_size * [sentence_max_length * [None]]
        for i, sentence in enumerate(metadata):
            for j, token in enumerate(sentence):
                wordform = token["form"]
                top_rule = top_rules[i][j]
                predictions[i][j] = self.predict_lemma_from_rule_id(wordform, top_rule)

        # If we are at inference, try to improve classifier predictions using
        # external dictionaries (if given).
        if not self.training:
            self.correct_predictions(logits, predictions, metadata)

        return {'preds': predictions, 'loss': loss}

    def correct_predictions(
        self,
        logits: Tensor,
        predictions: List[List[str]],
        metadata: Dict
    ):
        # If no dictionaries given, there's nothing to do.
        if not self.paradigm_dictionary and not self.dictionary_lemmas:
            return

        for i, sentence in enumerate(metadata):
            for j, token in enumerate(sentence):
                wordform = token["form"]

                # Lemmatizer usually does well with titles (e.g. 'Вася') and different kind of dates (like '70-е'),
                # whereas dictionaries don't, so skip any "corrections" in that case.
                if is_punctuation(wordform) or contains_digit(wordform):
                    continue

                # First check if word is present in dictionary of paradigms.
                # If it is, and it also has no collisions, then we can predict its lemma with 100% confidence.
                if self.paradigm_dictionary:
                    lemma = self.lookup_lemma_in_paradigm_dictionary(wordform, token["lemma"])
                    if lemma is not None:
                        predictions[i][j] = lemma
                        continue

                # Try to correct the answer within top-k predictions using external dictionary lemmas (if given).
                if self.dictionary_lemmas:
                    # Lemmatizes handles titles well, skip them as well.
                    if is_title(wordform):
                        continue
                    lemma = self.find_topk_dictionary_lemma(wordform, logits[i][j])
                    if lemma is not None:
                        predictions[i][j] = lemma
                        continue

                                                                # FIXME!
    def lookup_lemma_in_paradigm_dictionary(self, wordform: str, gold_lemma: str) -> Optional[str]:
        wordform_normalized = normalize(wordform)
        if wordform_normalized in self.paradigm_dictionary:
            lemma = self.paradigm_dictionary[wordform_normalized]
            if lemma != LemmaClassifier.COLLISION_TOKEN:
                if lemma != normalize(gold_lemma):
                    print(f"Compreno: token {wordform_normalized}, dictionary_lemma={lemma} != {normalize(gold_lemma)}=gold_lemma")
                return lemma
            else:
                pass
                # print(f"wordform_normalized: {wordform_normalized} has collision")
        return None

    def find_topk_dictionary_lemma(self, wordform: str, token_logits: Tensor) -> Optional[str]:
        topk_lemma_rules = torch.topk(token_logits, k=self.topk, dim=-1).indices.numpy()

        # Now find the most probable dictionary lemma for the word.
        for lemma_rule_id in topk_lemma_rules:
            lemma_normalized = self.predict_lemma_from_rule_id(wordform, lemma_rule_id)
            if lemma_normalized in self.dictionary_lemmas:
                return lemma_normalized
        return None

    def predict_lemma_from_rule_id(self, wordform: str, lemma_rule_id: int) -> str:
        lemma_rule_str = self.vocab.get_token_from_index(lemma_rule_id, "lemma_rule_labels")

        # Skip out-of-vocabulary words.
        if lemma_rule_str == DEFAULT_OOV_TOKEN:
            return DEFAULT_LEMMA_RULE

        lemma_rule = LemmaRule.from_str(lemma_rule_str)
        lemma = predict_lemma_from_rule(wordform, lemma_rule)
        lemma_normalized = normalize(lemma)
        return lemma_normalized

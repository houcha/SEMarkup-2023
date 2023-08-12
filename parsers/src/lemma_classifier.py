from overrides import override
from typing import Dict, List, Optional

import re
import string

import torch
from torch import Tensor

from allennlp.models import Model
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN

from .vocabulary import VocabularyWeighted
from .lemmatize_helper import LemmaRule, predict_lemma_from_rule, normalize, DEFAULT_LEMMA_RULE
from .feedforward_classifier import FeedForwardClassifier
from .accuracy import Accuracy


@Model.register('lemma_classifier')
class LemmaClassifier(Model):
    """
    FeedForwardClassifier specialization for lemma classification.
    """

    PUNCTUATION = set(string.punctuation)
    COLLISION_TOKEN = "@@COLLISION@@"

    def __init__(
        self,
        vocab: VocabularyWeighted,
        labels_namespace: str,
        in_dim: int,
        hid_dim: int,
        activation: str,
        dropout: float,
        paradigm_dictionary_path: str = None,
        dictionary_lemmas_info: List[Dict[str, str]] = [],
        topk: int = 1
    ):
        super().__init__(vocab)
        self.classifier = FeedForwardClassifier(
            vocab, labels_namespace, in_dim, hid_dim, activation, dropout
        )
        self.labels_namespace = labels_namespace

        # Paradigm dictionary.
        self.paradigm_dictionary = dict()
        if paradigm_dictionary_path:
            self.paradigm_dictionary = self.build_paradigm_dictionary(paradigm_dictionary_path)

        # Dictionary lemmas.
        self.dictionary_lemmas = set()
        for lemmas_info in dictionary_lemmas_info:
            self.dictionary_lemmas |= self.find_lemmas(lemmas_info["path"], lemmas_info["lemma_match_pattern"])
        assert topk == 1 if not self.dictionary_lemmas else topk >= 1
        self.topk = topk

        # Metric
        self.metric = Accuracy()

    @override(check_signature=False)
    def forward(
        self,
        embeddings: Tensor,
        labels: Tensor = None,
        mask: Tensor = None,
        metadata: Dict = None
    ) -> Dict[str, Tensor]:

        classifier_output = self.classifier.forward(embeddings, labels, mask)
        logits, loss = classifier_output['logits'].cpu(), classifier_output['loss']
        labels, mask = labels.cpu().numpy(), mask.cpu().numpy()

        batch_size, sentence_max_length = logits.shape[0], logits.shape[1]
        # Find top most confident lemma rules for each token.
        top_rules = torch.argmax(logits, dim=-1).numpy()
        # Predict lemmas.
        predictions = [[None for i in range(sentence_max_length)] for j in range(batch_size)]
        gold_lemmas = [[None for i in range(sentence_max_length)] for j in range(batch_size)]
        for i, sentence in enumerate(metadata):
            for j, token in enumerate(sentence):
                wordform = token["form"]
                predictions[i][j] = self.predict_lemma_from_rule_id(wordform, top_rules[i][j])
                # Also convert gold labels if provided.
                if labels is not None:
                    gold_lemmas[i][j] = normalize(token["lemma"])

        # If we are at inference, try to improve classifier predictions.
        if not self.training:
            self.correct_predictions(logits, predictions, metadata)

        if labels is not None:
            flatten = lambda list2d: [el for list1d in list2d for el in list1d]
            self.update_metrics(
                flatten(predictions),
                flatten(gold_lemmas),
                mask.flatten()
            )

        return {'preds': predictions, 'loss': loss}

    def update_metrics(self, predictions: List, labels: List, mask):
        self.metric(predictions, labels, mask)

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self.metric.get_metric(reset),
            "loss": self.classifier.get_metrics(reset)["loss"]
        }

    # ----------------
    # Private methods.
    # ----------------

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

    def correct_predictions(
        self,
        logits: Tensor,
        predictions: List[List[str]],
        metadata: Dict
    ):
        """
        Correct classifier predictions using external dictionaries (if given).
        """

        # If no dictionaries given, there's nothing to do.
        if not self.paradigm_dictionary and not self.dictionary_lemmas:
            return

        for i, sentence in enumerate(metadata):
            for j, token in enumerate(sentence):
                wordform = token["form"]

                # Lemmatizer usually does well with titles (e.g. 'Вася') and different kind of dates (like '70-е'),
                # whereas dictionaries don't, so skip any "corrections" in that case.
                if LemmaClassifier.is_punctuation(wordform) or LemmaClassifier.contains_digit(wordform):
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
                    if LemmaClassifier.is_title(wordform):
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
        lemma_rule_str = self.vocab.get_token_from_index(lemma_rule_id, self.labels_namespace)

        # Skip out-of-vocabulary words.
        if lemma_rule_str == DEFAULT_OOV_TOKEN:
            return DEFAULT_LEMMA_RULE

        lemma_rule = LemmaRule.from_str(lemma_rule_str)
        lemma = predict_lemma_from_rule(wordform, lemma_rule)
        lemma_normalized = normalize(lemma)
        return lemma_normalized

    @staticmethod
    def is_punctuation(word: str) -> bool:
        return word in LemmaClassifier.PUNCTUATION

    @staticmethod
    def contains_digit(word: str) -> bool:
        return any(char.isdigit() for char in word)

    @staticmethod
    def is_title(word: str) -> bool:
        return word[0].isupper()


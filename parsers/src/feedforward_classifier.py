from overrides import override
from typing import Dict, List

import re
import string

import torch
from torch import nn
from torch import Tensor

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.nn.activations import Activation
from allennlp.training.metrics import CategoricalAccuracy

from .lemmatize_helper import LemmaRule, predict_lemma_from_rule, normalize, DEFAULT_LEMMA_RULE


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
            self.paradigm_dictionary = self.build_paradigm_dictionary(paradigm_dictionary_path)

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
                paradigm_id, lemma = line.strip().split(':')
                paradigm_id, lemma_normalized = int(paradigm_id), normalize(lemma)

                # Use set to avoid duplicates within a paradigm.
                unique_wordforms = set()
                while (line := f.readline()) != '\n':
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
        logits, _, loss = output['logits'], output['preds'], output['loss']

        batch_size, sentence_max_length = logits.shape[0], logits.shape[1]
        # Allocate some space for answers.
        preds = [[None] * sentence_max_length] * batch_size
        # Find top most confident lemma rules for each token.
        top_rules = torch.topk(logits, k=self.topk, dim=-1).indices.cpu().numpy()

        for i, sentence in enumerate(metadata):
            for j, token in enumerate(sentence):
                wordform = token["form"]

                # Set the most probable lemma as a "preliminary" answer.
                token_top_rules = top_rules[i][j]
                token_top_rule = token_top_rules[0]
                lemma_rule_str = self.vocab.get_token_from_index(token_top_rule, "lemma_rule_labels")
                lemma_rule = DEFAULT_LEMMA_RULE if lemma_rule_str == DEFAULT_OOV_TOKEN else LemmaRule.from_str(lemma_rule_str)
                lemma = predict_lemma_from_rule(wordform, lemma_rule)
                lemma_normalized = normalize(lemma)
                preds[i][j] = lemma_normalized

                # If we are at inference, try to improve classifier predictions using external dictionaries (if given).
                if not self.training:
                    # First check if word is present in dictionary of paradigms.
                    # If it is, and it also has no collisions, then we can predict its lemma with 100% confidence.
                    if self.paradigm_dictionary:
                        wordform_normalized = normalize(wordform)
                        if wordform_normalized in self.paradigm_dictionary:
                            lemma = self.paradigm_dictionary[wordform_normalized]
                            if lemma != LemmaClassifier.COLLISION_TOKEN:
                                preds[i][j] = lemma
                                # FIXME
                                if labels is not None and lemma != normalize(token['lemma']):
                                    print(f"Compreno: token {wordform_normalized}, test_lemma={lemma} != {normalize(token['lemma'])}=gold_lemma")
                                # The answer is found, go to the next token.
                                continue

                    # Try to correct the answer within top-k predictions using external dictionary lemmas (if given).
                    if self.dictionary_lemmas:
                        # Lemmatizer usually does well with titles (e.g. 'Вася') and
                        # different kind of dates (like '70-е'), so we don't correct the predictions in that case.
                        is_punctuation = lambda word: word in LemmaClassifier.PUNCTUATION
                        is_title = lambda word: word[0].isupper()
                        contains_digit = lambda word: any(char.isdigit() for char in word)
                        if is_punctuation(wordform) or is_title(wordform) or contains_digit(wordform):
                            continue

                        print(f"Failed to find {wordform_normalized} in paradigm")

                        # Now find the most probable dictionary lemma for the word.
                        for lemma_rule_id in token_top_rules:
                            lemma_rule_str = self.vocab.get_token_from_index(lemma_rule_id, "lemma_rule_labels")

                            # Skip out-of-vocabulary words.
                            if lemma_rule_str == DEFAULT_OOV_TOKEN:
                                continue

                            lemma_rule = LemmaRule.from_str(lemma_rule_str)
                            lemma = predict_lemma_from_rule(wordform, lemma_rule)
                            lemma_normalized = normalize(lemma)
                            if lemma_normalized in self.dictionary_lemmas:
                                # If dictionary lemma encountered, update the answer and go to the next token.
                                preds[i][j] = lemma_normalized
                                break

        return {'preds': preds, 'loss': loss}


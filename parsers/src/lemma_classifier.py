from overrides import override
from typing import Dict, List, Optional

import re
import string
import ast

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
    Lemmas feed-forward classifier.
    """

    PUNCTUATION = set(string.punctuation)

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
            self.paradigm_dictionary = self.parse_paradigm_dictionary(paradigm_dictionary_path)

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
        metadata: Dict = None,
        feats_predictions = None
    ) -> Dict[str, Tensor]:

        classifier_output = self.classifier.forward(embeddings, labels, mask)
        logits, loss, mask = classifier_output['logits'].cpu(), classifier_output['loss'], mask.cpu().numpy()

        batch_size, sentence_max_length, _ = logits.shape
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
            self.correct_predictions(logits, predictions, metadata, feats_predictions)

        if labels is not None:
            flatten = lambda list2d: [el for list1d in list2d for el in list1d]
            self.update_metrics(
                flatten(predictions),
                flatten(gold_lemmas),
                mask.flatten()
            )

        return {'predictions': predictions, 'loss': loss}

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

    def parse_paradigm_dictionary(self, dictionary_path: str) -> dict:
        dictionary = dict()
        with open(dictionary_path, 'r') as f:
            while line := f.readline():
                # Skip newlines.
                if not line.strip():
                    continue

                lemma, lemma_feats = line.strip().split('\t')
                lemma_normalized = normalize(lemma)
                lemma_feats_ud = self.convert_msd_to_ud(lemma_feats, lemma_normalized)

                # Use set to avoid duplicates within a paradigm.
                wordforms_feats = dict()
                wordforms_feats[lemma_normalized] = lemma_feats_ud
                while (line := f.readline()):
                    # Newline indicates the end of the paradigm group.
                    if not line.strip():
                        break
                    wordform, msd_feats = line.strip().split('\t')
                    wordform_normalized = normalize(wordform)
                    ud_feats = self.convert_msd_to_ud(msd_feats, wordform_normalized)
                    #if ud_feats:
                    #    print(f"wordform: {wordform_normalized} msd_feats: {msd_feats} ud_feats: {ud_feats}")
                    #if wordform_normalized == 'ком':
                    #    print(f"msd_feats: {msd_feats}, ud_feats: {ud_feats}")
                    wordforms_feats[wordform_normalized] = ud_feats

                for wordform, feats in wordforms_feats.items():
                    if wordform not in dictionary:
                        dictionary[wordform] = dict()
                    # If there is more than one lemma for the wordform and the features in the dictionary
                    # then paradigm dictionary is of no use.
                    if str(feats) in dictionary[wordform] and lemma_normalized != dictionary[wordform][str(feats)]:
                        # print(f"== word {wordform}, {str(feats)} and lemma {lemma_normalized} already present with lemma {dictionary[wordform][str(feats)]}=already in dict")
                        dictionary[wordform][str(feats)] = None
                    else:
                        dictionary[wordform][str(feats)] = lemma_normalized

        # Collapse single element features in order to reduce allocated space.
        for wordform, feats in dictionary.items():
            lemmas = set()
            for feat, lemma in feats.items():
                lemmas.add(lemma)
            if len(lemmas) == 1:
                dictionary[wordform] = list(lemmas)[0]
        return dictionary

    def correct_predictions(
        self,
        logits: Tensor,
        predictions: List[List[str]],
        metadata: Dict,
        feats_predictions = None
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
                # FIXME
                gold_lemma = token["lemma"]

                # Lemmatizer usually does well with titles (e.g. 'Вася') and different kind of dates (like '70-е'),
                # whereas dictionaries don't, so skip any "corrections" in that case.
                if LemmaClassifier.is_punctuation(wordform) or LemmaClassifier.contains_digit(wordform):
                    continue

                # First check if word is present in dictionary of paradigms.
                # If it is, and it also has no collisions, then we can predict its lemma with 100% confidence.
                if self.paradigm_dictionary:
                    lemma = self.lookup_lemma_in_paradigm_dictionary(wordform, feats_predictions[i][j], gold_lemma)
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

    def lookup_lemma_in_paradigm_dictionary(
        self,
        wordform: str,
        feats_predicted: str,
        gold_lemma # FIXME
    ) -> Optional[str]:

        def convert_str_feats_to_dict(feats: str) -> dict:
            feats_dict = dict()
            for feat in feats.split('|'):
                try:
                    category, grammeme = feat.split('=')
                    assert category not in feats_dict
                    feats_dict[category] = grammeme
                except:
                    continue
            return feats_dict

        result_lemma = None
        wordform_normalized = normalize(wordform)
        if wordform_normalized in self.paradigm_dictionary:
            value = self.paradigm_dictionary[wordform_normalized]
            if isinstance(value, dict):
                feats_to_lemma = value

                # FIXME
                has_gold_lemma_in_featues = False
                lemmas_tried = []

                feats_predicted = convert_str_feats_to_dict(feats_predicted)

                best_similarity, best_lemma = 0.0, None
                for feats, lemma_normalized in feats_to_lemma.items():
                    feats = ast.literal_eval(feats) # Convert sting back to dict.
                    similarity = self.count_common_features(feats, feats_predicted)
                    # FIXME
                    if lemma_normalized == normalize(gold_lemma):
                        has_gold_lemma_in_featues = True
                    lemmas_tried.append(lemma_normalized)
                    if best_similarity < similarity:
                        best_similarity = similarity
                        best_lemma = lemma_normalized
                result_lemma = best_lemma
                # FIXME
                #if not has_gold_lemma_in_featues:
                #    print(f"Compreno: token {wordform_normalized}, dictionary_lemmas={feats_to_lemma.items()} != {normalize(gold_lemma)}=gold_lemma")
            else:
                # Single lemma, no collisions.
                lemma = value
                result_lemma = lemma
                #if lemma != normalize(gold_lemma):
                #    print(f"Compreno: token {wordform_normalized}, dictionary_lemma={lemma} != {normalize(gold_lemma)}=gold_lemma")
        return result_lemma

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

    @staticmethod
    def convert_msd_to_ud(msd: str, wordform: str # FIXME
    ) -> dict:
        convertion_table = [
            { # 0
                "Noun":      ("pos", "NOUN"),
                "Adjective": ("pos", "ADJ"),
                "Adverb":    ("pos", "ADV"),
                "Verb":      ("pos", "VERB"),
            },
            { # 1
                "GTImperative":            ("Mood", "Imp"),
                "GTVerb":                  ("VerbForm", "Fin"),
                "GTInfinitive":            ("VerbForm", "Inf"),
                "GTParticiple":            ("VerbForm", "Part"),
                "GTParticipleAttributive": ("VerbForm", "Part"),
                "GTAdverb":                ("VerbForm", "Conv"),
            },
            {},# 2
            { # 3
                "Animate":   ("Animacy", "Anim"),
                "Inanimate": ("Animacy", "Inan")
            },
            { # 4
                "Imperfective": ("Aspect", "Imp"),
                "Perfective":   ("Aspect", "Perf")
            },
            { # 5
                "Nominative":    ("Case", "Nom"),
                "Genitive":      ("Case", "Gen"),
                "Partitive":     ("Case", "Par"),
                "Dative":        ("Case", "Dat"),
                "Accusative":    ("Case", "Acc"),
                "Prepositional": ("Case", "Loc"),
                "Locative":      ("Case", "Loc"),
                "Instrumental":  ("Case", "Ins"),
                "Vocative":      ("Case", "Voc"),
            },
            {},# 6
            {},# 7
            {},# 8
            {},# 9
            {
                "PluraliaTantum":   ("Number", "Plur"),
                "SingulariaTantum": ("Number", "Sing")
            },# 10
            {},# 11
            { # 12
                "DegreePositive":    ("Degree", "Pos"),
                "DegreeComparative": ("Degree", "Cmp"),
                "DegreeSuperlative": ("Degree", "Sup")
            },
            {},# 13
            { # 14
                "Masculine": ("Gender", "Masc"),
                "Feminine":  ("Gender", "Fem"),
                "Neuter":    ("Gender", "Neut")
            },
            {},# 15
            {},# 16
            { # 17
                "Singular": ("Number", "Sing"),
                "Plural":   ("Number", "Plur")
            },
            {},# 18
            {},# 19
            {},# 20
            { # 21
                "PersonFirst":  ("Person", 1),
                "PersonSecond": ("Person", 2),
                "PersonThird":  ("Person", 3),
            },
            {},# 22
            {},# 23
            {},# 24
            {},# 25
            {},# 26
            {},# 27
            { # 28
                "Past":    ("Tense", "Past"),
                "Present": ("Tense", "Pres"),
                "Future":  ("Tense", "Fut"),
            },
            {}, # 29
            { # 30
                "Active":   ("Voice", "Act"),
                "Passive":  ("Voice", "Pass"),
                "VoiceSya": ("Voice", "Mid")
            }
        ]
        ud_feats = dict()
        msd_feats = msd.split(';')
        pos = msd_feats[0]
        if pos in convertion_table[0]:
            for position, grammeme in enumerate(msd_feats):
                if position == 0:
                    assert grammeme in convertion_table[0]
                if not grammeme or not convertion_table[position]:
                    continue
                if grammeme not in convertion_table[position]:
                   #assert msd.split(';')[0] in {"Noun", "Adjective", "Adverb", "Verb"}, f"{msd.split(';')[0]} is extra tag"
                   print(f"{grammeme} at {position} position is OOV, allowed grammemes: {convertion_table[position].keys()}, wordform: {wordform}, msd feats: {msd}")
                   continue
                ud_category, ud_tag = convertion_table[position][grammeme]
                ud_feats[ud_category] = ud_tag
        return ud_feats

    @staticmethod
    def count_common_features(
        lhs_feats: Dict[str, str],
        rhs_feats: Dict[str, str]
    ) -> float:
        common_feats_count = sum([
            lhs_feats[category] == rhs_feats[category]
            for category in lhs_feats
            if category in rhs_feats
        ])
        return common_feats_count


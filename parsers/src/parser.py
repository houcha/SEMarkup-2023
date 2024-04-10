from overrides import override
from typing import Dict, List

import numpy as np

import torch
from torch import Tensor, BoolTensor

from allennlp.nn.util import get_text_field_mask, move_to_device, get_device_of
from allennlp.common import Lazy
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.data.token_indexers import TokenIndexer

import sys
sys.path.append("..")
from common.token import Token

from .feedforward_classifier import FeedForwardClassifier
from .lemma_classifier import LemmaClassifier
from .lemmatize_helper import LemmaRule, predict_lemma_from_rule
from .null_classifier import NullClassifier
from .dependency_classifier import DependencyClassifier
from .utils import get_null_mask


@Model.register('morpho_syntax_semantic_parser')
class MorphoSyntaxSemanticParser(Model):
    """
    Joint Morpho-Syntax-Semantic Parser.
    See https://guide.allennlp.org/your-first-model for guidance.
    """

    # See https://guide.allennlp.org/using-config-files to find more about Lazy.
    #
    # TODO: move Lazy to from_lazy_objects (as here https://guide.allennlp.org/using-config-files#4)
    def __init__(
        self,
        vocab: Vocabulary,
        indexer: TokenIndexer,
        embedder: TokenEmbedder,
        lemma_rule_classifier: Lazy[LemmaClassifier],
        pos_feats_classifier: Lazy[FeedForwardClassifier],
        depencency_classifier: Lazy[DependencyClassifier],
        misc_classifier: Lazy[FeedForwardClassifier],
        semslot_classifier: Lazy[FeedForwardClassifier],
        semclass_classifier: Lazy[FeedForwardClassifier],
        null_classifier: Lazy[NullClassifier]
    ):
        super().__init__(vocab)

        self.embedder = embedder
        embedding_dim = self.embedder.get_output_dim()

        self.lemma_rule_classifier = lemma_rule_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("lemma_rule_labels"),
        )
        self.pos_feats_classifier = pos_feats_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("pos_feats_labels"),
        )
        self.dependency_classifier = depencency_classifier.construct(
            in_dim=embedding_dim,
            n_rel_classes_ud=vocab.get_vocab_size("deprel_labels"),
            n_rel_classes_eud=vocab.get_vocab_size("deps_labels"),
        )
        self.misc_classifier = misc_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("misc_labels"),
        )
        self.semslot_classifier = semslot_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semslot_labels"),
        )
        self.semclass_classifier = semclass_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semclass_labels"),
        )
        self.null_classifier = null_classifier.construct(
            indexer=indexer,
            embedder=embedder,
            in_dim=embedding_dim,
        )

    @override(check_signature=False)
    def forward(
        self,
        words: TextFieldTensors,
        sentences: List[List[Token]],
        lemma_rule_labels: Tensor = None,
        pos_feats_labels: Tensor = None,
        deprel_labels: Tensor = None,
        deps_labels: Tensor = None,
        misc_labels: Tensor = None,
        semslot_labels: Tensor = None,
        semclass_labels: Tensor = None,
        metadata: Dict = None
    ) -> Dict[str, Tensor]:
        # If all labels are empty, we are at inference.
        is_inference = lemma_rule_labels is None \
            and pos_feats_labels is None \
            and deprel_labels is None \
            and deps_labels is None \
            and misc_labels is None  \
            and semslot_labels is None \
            and semclass_labels is None \

        device = get_device_of(words["tokens"]["token_ids"])

        words_with_nulls, sentences_with_nulls, null_loss = self.null_classifier(words, sentences, is_inference)

        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedder(**words_with_nulls['tokens'])
        # Padding mask.
        mask = get_text_field_mask(words_with_nulls)
        # Mask with nulls excluded.
        null_mask = get_null_mask(sentences_with_nulls)
        null_mask = move_to_device(null_mask, device)

        # Mask nulls, since they have trivial lemmas.
        lemma_rule = self.lemma_rule_classifier(embeddings, lemma_rule_labels, mask & (~null_mask), metadata)
        # Don't mask nulls, as they actually have non-trivial grammatical features we want to learn.
        pos_feats = self.pos_feats_classifier(embeddings, pos_feats_labels, mask)
        # Mask nulls for basic UD and don't mask for E-UD.
        syntax = self.dependency_classifier(embeddings, deprel_labels, deps_labels, mask & (~null_mask), mask)
        #print(misc_labels)
        misc = self.misc_classifier(embeddings, misc_labels, mask)
        semslot = self.semslot_classifier(embeddings, semslot_labels, mask)
        semclass = self.semclass_classifier(embeddings, semclass_labels, mask)

        loss = lemma_rule['loss'] \
            + pos_feats['loss'] \
            + syntax['arc_loss_ud'] \
            + syntax['rel_loss_ud'] \
            + syntax['arc_loss_eud'] \
            + syntax['rel_loss_eud'] \
            + misc['loss'] \
            + semslot['loss'] \
            + semclass['loss'] \
            + null_loss

        return {
            'sentences': sentences_with_nulls,
            'lemma_rule_preds': lemma_rule['preds'],
            'pos_feats_preds': pos_feats['preds'],
            'syntax_ud': syntax['syntax_ud'],
            'syntax_eud': syntax['syntax_eud'],
            'misc_preds': misc['preds'],
            'semslot_preds': semslot['preds'],
            'semclass_preds': semclass['preds'],
            'loss': loss,
            'metadata': metadata,
        }

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Morphology.
        lemma_accuracy = self.lemma_rule_classifier.get_metrics(reset)['Accuracy']
        pos_feats_accuracy = self.pos_feats_classifier.get_metrics(reset)['Accuracy']
        # Syntax.
        syntax_metrics = self.dependency_classifier.get_metrics(reset)
        uas_ud = syntax_metrics['UD-UAS']
        las_ud = syntax_metrics['UD-LAS']
        uas_eud = syntax_metrics['EUD-UAS']
        las_eud = syntax_metrics['EUD-LAS']
        # Misc
        misc_accuracy = self.misc_classifier.get_metrics(reset)['Accuracy']
        # Semantic.
        semslot_accuracy = self.semslot_classifier.get_metrics(reset)['Accuracy']
        semclass_accuracy = self.semclass_classifier.get_metrics(reset)['Accuracy']
        # Average.
        mean_accuracy = np.mean([
            lemma_accuracy,
            pos_feats_accuracy,
            uas_ud,
            las_ud,
            uas_eud,
            las_eud,
            misc_accuracy,
            semslot_accuracy,
            semclass_accuracy
        ])
        # Nulls (do not average).
        null_metrics = self.null_classifier.get_metrics(reset)
        null_accuracy = null_metrics["Accuracy"]
        null_f1score = null_metrics["f1"]

        return {
            'NullAccuracy': null_accuracy,
            'NullF1': null_f1score,
            'Lemma': lemma_accuracy,
            'PosFeats': pos_feats_accuracy,
            'UD-UAS': uas_ud,
            'UD-LAS': las_ud,
            'EUD-UAS': uas_eud,
            'EUD-LAS': las_eud,
            'Misc': misc_accuracy,
            'SS': semslot_accuracy,
            'SC': semclass_accuracy,
            'Avg': mean_accuracy
        }

    @override(check_signature=False)
    def make_output_human_readable(self, output: Dict[str, Tensor]) -> Dict[str, list]:
        sentences = output["sentences"]
        # Make sure batch_size is 1 during prediction
        assert len(sentences) == 1
        sentence = sentences[0]

        # Restore ids.
        ids = [token.id for token in sentence]
        # Restore forms.
        forms = [token.form for token in sentence]

        # Restore lemmas.
        lemmas = []
        for token, lemma_rule_pred in zip(sentence, output["lemma_rule_preds"].tolist()[0]):
            word = token.form
            lemma_rule_str = self.vocab.get_token_from_index(lemma_rule_pred, "lemma_rule_labels")
            if lemma_rule_str == DEFAULT_OOV_TOKEN:
                lemma = '_'
            else:
                lemma_rule = LemmaRule.from_str(lemma_rule_str)
                lemma = predict_lemma_from_rule(word, lemma_rule)
            lemmas.append(lemma)

        # Restore "glued" pos and feats tags.
        upos_tags = []
        xpos_tags = []
        feats_tags = []
        for pos_feats_pred in output["pos_feats_preds"].tolist()[0]:
            pos_feats_str = self.vocab.get_token_from_index(pos_feats_pred, "pos_feats_labels")
            if pos_feats_str == DEFAULT_OOV_TOKEN:
                upos_tag, xpos_tag, feats_tag = '_', '_', '_'
            else:
                upos_tag, xpos_tag, feats_tag = pos_feats_str.split('#')
            upos_tags.append(upos_tag)
            xpos_tags.append(xpos_tag)
            feats_tags.append(feats_tag)

        # Restore heads and deprels.
        # Recall that syntactic nodes are renumerated so that [1, 1.1, 2] becomes [1, 2, 3].
        # Now we want to bind them back to the actual tokens, i.e. renumerate [1, 2, 3] into [1, 1.1, 2].
        # Luckily, we already have this mapping stored in 'ids'.
        heads = [None for _ in range(len(sentence))]
        deprels = [None for _ in range(len(sentence))]
        for batch_index, edge_to, edge_from, deprel_id in output["syntax_ud"].tolist():
            # Make sure UD-heads are unique (have no collisions).
            assert heads[edge_to] is None
            assert deprels[edge_to] is None
            # Renumerate nodes back to actual tokens' ids and replace self-loops with ROOT (0).
            heads[edge_to] = ids[edge_from] if edge_from != edge_to else 0
            deprels[edge_to] = self.vocab.get_token_from_index(deprel_id, "deprel_labels")

        # Restore deps.
        deps = [[] for _ in range(len(sentence))]
        for batch_index, edge_to, edge_from, dep_id in output["syntax_eud"].tolist():
            dep = self.vocab.get_token_from_index(dep_id, "deps_labels")
            # Renumerate nodes back to actual tokens' ids and replace self-loops with ROOT (0).
            deps[edge_to].append(f"{ids[edge_from] if edge_from != edge_to else 0}:{dep}")
        deps = ['|'.join(dep) if dep else '_' for dep in deps]

        # Restore miscs.
        misc_preds = output["misc_preds"].tolist()[0]
        miscs = [self.vocab.get_token_from_index(misc, "misc_labels") for misc in misc_preds]

        # Restore semslots.
        semslot_preds = output["semslot_preds"].tolist()[0]
        semslots = [self.vocab.get_token_from_index(semslot, "semslot_labels")
                    if semslot != DEFAULT_OOV_TOKEN else '_'
                    for semslot in semslot_preds]

        # Restore semclasses.
        semclass_preds = output["semclass_preds"].tolist()[0]
        semclasses = [self.vocab.get_token_from_index(semclass, "semclass_labels")
                      if semclass != DEFAULT_OOV_TOKEN else '_'
                      for semclass in semclass_preds]

        metadata = output["metadata"][0]

        # Manually post-process nulls' tags.
        for i, token in enumerate(sentence):
            if token.is_null():
                heads[i] = '_'
                deprels[i] = '_'
                miscs[i] = 'ellipsis'

        return {
            "ids": [ids],
            "forms": [forms],
            "lemmas": [lemmas],
            "upos": [upos_tags],
            "xpos": [xpos_tags],
            "feats": [feats_tags],
            "heads": [heads],
            "deprels": [deprels],
            "deps": [deps],
            "miscs": [miscs],
            "semslots": [semslots],
            "semclasses": [semclasses],
            "metadata": [metadata],
        }

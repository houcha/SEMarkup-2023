# from overrides import override

from typing import Dict

import numpy as np

import torch
from torch import Tensor, BoolTensor

from allennlp.nn.util import get_text_field_mask
from allennlp.common import Lazy
from allennlp.data import TextFieldTensors
from allennlp.data.fields import TensorField
from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

from .feedforward_classifier import FeedForwardClassifier, LemmaClassifier
from .dependency_classifier import DependencyClassifier
from .lemmatize_helper import LemmaRule, predict_lemma_from_rule


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
        embedder: TokenEmbedder,
        lemma_rule_classifier: Lazy[LemmaClassifier],
        pos_feats_classifier: Lazy[FeedForwardClassifier],
        depencency_classifier: Lazy[DependencyClassifier],
        semslot_classifier: Lazy[FeedForwardClassifier],
        semclass_classifier: Lazy[FeedForwardClassifier],
        null_classifier: Lazy[FeedForwardClassifier]
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
        self.semslot_classifier = semslot_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semslot_labels"),
        )
        self.semclass_classifier = semclass_classifier.construct(
            in_dim=embedding_dim,
            n_classes=vocab.get_vocab_size("semclass_labels"),
        )
        self.null_classifier = null_classifier.construct(
            in_dim=embedding_dim,
            n_classes=2
        )

    # @override(check_signature=False)
    def forward(
        self,
        words: TextFieldTensors,
        null_mask: Tensor,
        words_nulls_excluded: TextFieldTensors,
        lemma_rule_labels: Tensor = None,
        pos_feats_labels: Tensor = None,
        deprel_labels: Tensor = None,
        deps_labels: Tensor = None,
        misc_labels: Tensor = None,
        semslot_labels: Tensor = None,
        semclass_labels: Tensor = None,
        metadata: Dict = None
    ) -> Dict[str, Tensor]:

        # [batch_size, seq_len]
        mask_nulls_included = get_text_field_mask(words)
        mask_nulls_excluded = get_text_field_mask(words_nulls_excluded)
        # [batch_size, seq_len]
        target_has_null_after = self._build_has_null_after(null_mask, mask_nulls_included).long()
        # Make sure reconstructed targets are of correct length.
        assert target_has_null_after.shape[1] == mask_nulls_excluded.shape[1]
        # [batch_size, seq_len, embedding_dim]
        embeddings_no_nulls = self.embedder(**words_nulls_excluded['tokens'])
        nulls = self.null_classifier(embeddings_no_nulls, target_has_null_after, mask_nulls_excluded)

        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedder(**words['tokens'])
        # [batch_size, seq_len]
        mask = get_text_field_mask(words)
        # Mask with nulls excluded.
        no_null_mask = (~null_mask)

        # Mask nulls, since they have trivial lemmas.
        lemma_rule = self.lemma_rule_classifier(embeddings, lemma_rule_labels, mask & no_null_mask, metadata)
        # Don't mask nulls, as they actually have non-trivial grammatical features we want to learn.
        pos_feats = self.pos_feats_classifier(embeddings, pos_feats_labels, mask)
        syntax = self.dependency_classifier(embeddings, deprel_labels, deps_labels, mask & no_null_mask, mask)
        semslot = self.semslot_classifier(embeddings, semslot_labels, mask)
        semclass = self.semclass_classifier(embeddings, semclass_labels, mask)

        loss = lemma_rule['loss'] \
            + pos_feats['loss'] \
            + syntax['arc_loss_ud'] \
            + syntax['rel_loss_ud'] \
            + syntax['arc_loss_eud'] \
            + syntax['rel_loss_eud'] \
            + semslot['loss'] \
            + semclass['loss'] \
            + nulls['loss']

        return {
            'lemma_rule_preds': lemma_rule['preds'],
            'pos_feats_preds': pos_feats['preds'],
            'head_preds': syntax['arc_preds'],
            'deprel_preds': syntax['rel_preds'],
            'semslot_preds': semslot['preds'],
            'semclass_preds': semclass['preds'],
            'loss': loss,
            'metadata': metadata,
        }

    @staticmethod
    def _build_has_null_after(null_mask: BoolTensor, padding_mask: BoolTensor) -> BoolTensor:
        """
        Return mask without nulls, where mask[:, i] = True iff token i has null after him in original sentence (with nulls).
        """
        has_null_after = null_mask.roll(shifts=-1, dims=-1)
        # If null is the first token in a sentence, then... well, we ignore it :|
        has_null_after[:, -1] = False
        not_null_mask = (~null_mask) & padding_mask
        non_null_tokens_count = not_null_mask.sum(1)
        has_null_after_trimmed = torch.masked_select(has_null_after, not_null_mask).split(non_null_tokens_count.tolist())
        return torch.nn.utils.rnn.pad_sequence(has_null_after_trimmed, batch_first=True, padding_value=False)

    # @override
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
            semslot_accuracy,
            semclass_accuracy
        ])
        # Nulls (do not average).
        null_accuracy = self.null_classifier.get_metrics(reset)['Accuracy']

        return {
            'Lemma': lemma_accuracy,
            'PosFeats': pos_feats_accuracy,
            'UD-UAS': uas_ud,
            'UD-LAS': las_ud,
            'EUD-UAS': uas_eud,
            'EUD-LAS': las_eud,
            'SS': semslot_accuracy,
            'SC': semclass_accuracy,
            'Avg': mean_accuracy,
            'Null': null_accuracy
        }

    # @override(check_signature=False)
    def make_output_human_readable(self, output: Dict[str, Tensor]) -> Dict[str, list]:
        sentences = output["metadata"]
        # Make sure batch_size is 1 during prediction, because 
        assert len(sentences) == 1
        sentence = sentences[0]
        metadata = sentence.metadata

        # Restore ids.
        ids = []
        for token in sentence:
            ids.append(token["id"])

        # Restore forms.
        forms = []
        for token in sentence:
            forms.append(token["form"])

        # Restore lemmas.
        lemmas = []
        lemma_rule_preds = output["lemma_rule_preds"].tolist()[0]
        for token, lemma_rule_pred in zip(sentence, lemma_rule_preds):
            word = token["form"]
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
        pos_feats_preds = output["pos_feats_preds"].tolist()[0]
        for pos_feats_pred in pos_feats_preds:
            pos_feats_str = self.vocab.get_token_from_index(pos_feats_pred, "pos_feats_labels")
            if pos_feats_str == DEFAULT_OOV_TOKEN:
                upos_tag, xpos_tag, feats_tag = '_', '_', '_'
            else:
                upos_tag, xpos_tag, feats_tag = pos_feats_str.split('#')
            upos_tags.append(upos_tag)
            xpos_tags.append(xpos_tag)
            feats_tags.append(feats_tag)

        # Restore heads.
        # Heads are integers, so simply convert them to strings.
        heads = list(map(str, output["head_preds"].tolist()[0]))

        # Restore deprels.
        deprels = []
        deprel_preds = output["deprel_preds"].tolist()[0]
        for deprel_pred in deprel_preds:
            deprel = self.vocab.get_token_from_index(deprel_pred, "deprel_labels")
            if deprel == DEFAULT_OOV_TOKEN:
                deprel = '_'
            deprels.append(deprel)

        # Restore semslots.
        semslots = []
        semslot_preds = output["semslot_preds"].tolist()[0]
        for semslot_pred in semslot_preds:
            semslot = self.vocab.get_token_from_index(semslot_pred, "semslot_labels")
            if semslot == DEFAULT_OOV_TOKEN:
                semslots = '_'
            semslots.append(semslot)

        # Restore semclasses.
        semclasses = []
        semclass_preds = output["semclass_preds"].tolist()[0]
        for semclass_pred in semclass_preds:
            semclass = self.vocab.get_token_from_index(semclass_pred, "semclass_labels")
            if semclass == DEFAULT_OOV_TOKEN:
                semclasss = '_'
            semclasses.append(semclass)

        return {
            "metadata": [metadata],
            "ids": [ids],
            "forms": [forms],
            "lemmas": [lemmas],
            "upos": [upos_tags],
            "xpos": [xpos_tags],
            "feats": [feats_tags],
            "heads": [heads],
            "deprels": [deprels],
            "semslots": [semslots],
            "semclasses": [semclasses],
        }


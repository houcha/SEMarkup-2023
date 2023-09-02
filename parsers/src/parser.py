from overrides import override

from typing import Dict

import numpy as np
from torch import Tensor

from allennlp.common import Lazy
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.initializers import InitializerApplicator

from .vocabulary import VocabularyWeighted
from .feedforward_classifier import FeedForwardClassifier
from .lemma_classifier import LemmaClassifier
from .dependency_classifier import DependencyClassifier


@Model.register('morpho_syntax_semantic_parser', constructor="from_lazy_objects")
class MorphoSyntaxSemanticParser(Model):
    """
    Joint Morpho-Syntax-Semantic Parser.
    See https://guide.allennlp.org/your-first-model for guidance.
    """

    def __init__(
        self,
        vocab: VocabularyWeighted,
        embedder: TokenEmbedder,
        lemma_rule_classifier: LemmaClassifier,
        feats_classifier: FeedForwardClassifier,
        dependency_classifier: DependencyClassifier,
        semslot_classifier: FeedForwardClassifier,
        semclass_classifier: FeedForwardClassifier,
        initializer: InitializerApplicator
    ):
        super().__init__(vocab)

        self.embedder = embedder
        self.lemma_rule_classifier = lemma_rule_classifier
        self.feats_classifier = feats_classifier
        self.dependency_classifier = dependency_classifier
        self.semslot_classifier = semslot_classifier
        self.semclass_classifier = semclass_classifier
        initializer(self)

    @classmethod
    def from_lazy_objects(
        cls,
        vocab: VocabularyWeighted,
        embedder: TokenEmbedder,
        lemma_rule_classifier: Lazy[LemmaClassifier],
        feats_classifier: Lazy[FeedForwardClassifier],
        dependency_classifier: Lazy[DependencyClassifier],
        semslot_classifier: Lazy[FeedForwardClassifier],
        semclass_classifier: Lazy[FeedForwardClassifier],
        initializer: InitializerApplicator = InitializerApplicator()
    ) -> "MorphoSyntaxSemanticParser":

        embedding_dim = embedder.get_output_dim()

        # Classifier are Lazy because they depend on embedder's output dimentions.
        lemma_rule_classifier_ = lemma_rule_classifier.construct(
            in_dim=embedding_dim,
            labels_namespace="lemma_rule_labels",
        )
        feats_classifier_ = feats_classifier.construct(
            in_dim=embedding_dim,
            labels_namespace="feats_labels",
        )
        dependency_classifier_ = dependency_classifier.construct(
            in_dim=embedding_dim,
            labels_namespace="deprel_labels",
        )
        semslot_classifier_ = semslot_classifier.construct(
            in_dim=embedding_dim,
            labels_namespace="semslot_labels",
        )
        semclass_classifier_ = semclass_classifier.construct(
            in_dim=embedding_dim,
            labels_namespace="semclass_labels",
        )
        return cls(
            vocab=vocab,
            embedder=embedder,
            lemma_rule_classifier=lemma_rule_classifier_,
            feats_classifier=feats_classifier_,
            dependency_classifier=dependency_classifier_,
            semslot_classifier=semslot_classifier_,
            semclass_classifier=semclass_classifier_,
            initializer=initializer
        )

    @override(check_signature=False)
    def forward(
        self,
        words: TextFieldTensors,
        lemma_rule_labels: Tensor = None,
        feats_labels: Tensor = None,
        head_labels: Tensor = None,
        deprel_labels: Tensor = None,
        semslot_labels: Tensor = None,
        semclass_labels: Tensor = None,
        metadata: Dict = None
    ) -> Dict[str, Tensor]:

        # [batch_size, seq_len, embedding_dim]
        embeddings = self.embedder(**words['tokens'])
        # [batch_size, seq_len]
        mask = get_text_field_mask(words)

        feats_output = self.feats_classifier(embeddings, feats_labels, mask)
        feats_predictions = None
        if not self.training:
            feats_predictions = self.feats_classifier.decode_ids(feats_output["prediction_ids"])
        lemma_output = self.lemma_rule_classifier(embeddings, lemma_rule_labels, mask, metadata, feats_predictions)
        syntax_output = self.dependency_classifier(embeddings, head_labels, deprel_labels, mask)
        semslot_output = self.semslot_classifier(embeddings, semslot_labels, mask)
        semclass_output = self.semclass_classifier(embeddings, semclass_labels, mask)

        loss = lemma_output['loss'] + \
               feats_output['loss'] + \
               syntax_output['arc_loss'] + \
               syntax_output['rel_loss'] + \
               semslot_output['loss'] + \
               semclass_output['loss']

        return {
            'lemma_predictions': lemma_output['predictions'],
            'feats_prediction_ids': feats_output['prediction_ids'],
            'head_predictions': syntax_output['arc_predictions'],
            'deprel_prediction_ids': syntax_output['rel_prediction_ids'],
            'semslot_prediction_ids': semslot_output['prediction_ids'],
            'semclass_prediction_ids': semclass_output['prediction_ids'],
            'loss': loss,
            'metadata': metadata,
        }

    @override
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Morphology.
        ## lemma
        lemma_metrics = self.lemma_rule_classifier.get_metrics(reset)
        lemma_accuracy = lemma_metrics['accuracy']
        lemma_loss = lemma_metrics['loss']
        ## pos
        feats_metrics = self.feats_classifier.get_metrics(reset)
        feats_accuracy = feats_metrics['accuracy']
        feats_macro_fscore = feats_metrics['macro-fscore']
        feats_loss = feats_metrics['loss']
        # Syntax.
        syntax_metrics = self.dependency_classifier.get_metrics(reset)
        uas = syntax_metrics['UAS']
        arc_loss = syntax_metrics["arc-loss"]
        las = syntax_metrics['LAS']
        deprel_loss = syntax_metrics["rel-loss"]
        # Semantic.
        ## semslot
        semslot_metrics = self.semslot_classifier.get_metrics(reset)
        semslot_accuracy = semslot_metrics['accuracy']
        semslot_macro_fscore = semslot_metrics['macro-fscore']
        semslot_loss = semslot_metrics['loss']
        ## semclass
        semclass_metrics = self.semclass_classifier.get_metrics(reset)
        semclass_accuracy = semclass_metrics['accuracy']
        semclass_macro_fscore = semclass_metrics['macro-fscore']
        semclass_loss = semclass_metrics['loss']
        # Average.
        mean_accuracy = np.mean([
            lemma_accuracy,
            feats_accuracy,
            uas,
            las,
            semslot_accuracy,
            semclass_accuracy
        ])

        return {
            'LemmaAccuracy': lemma_accuracy,
            'LemmaLoss': lemma_loss,
            'PosFeatsAccuracy': feats_accuracy,
            'PosFeatsMacroF1': feats_macro_fscore,
            'PosFeatsLoss': feats_loss,
            'UAS': uas,
            'HeadLoss': arc_loss,
            'LAS': las,
            'DeprelLoss': deprel_loss,
            'SemSlotAccuracy': semslot_accuracy,
            'SemSlotMacroF1': semslot_macro_fscore,
            'SemSlotLoss': semslot_loss,
            'SemClassAccuracy': semclass_accuracy,
            'SemClassMacroF1': semclass_macro_fscore,
            'SemClassLoss': semclass_loss,
            'AverageAccuracy': mean_accuracy,
        }

    @override(check_signature=False)
    def make_output_human_readable(self, output: Dict[str, Tensor]) -> Dict[str, list]:
        sentences = output["metadata"]

        # Restore ids and forms.
        ids = []
        forms = []
        for sentence in sentences:
            sentence_ids = []
            sentence_forms = []
            for token in sentence:
                sentence_ids.append(token["id"])
                sentence_forms.append(token["form"])
            ids.append(sentence_ids)
            forms.append(sentence_forms)

        # Lemma classifier handles predictions itself.
        lemmas = output["lemma_predictions"]

        # Restore "glued" POS and features tags.
        pos_tags = []
        feats_tags = []
        pos_feats_predictions = self.feats_classifier.decode_ids(output["feats_prediction_ids"])
        for sentence_pos_feats_predictions in pos_feats_predictions:
            sentence_pos_tags = []
            sentence_feats_tags = []
            for pos_feats_prediction in sentence_pos_feats_predictions:
                pos_tag, feats_tag = '_', '_'
                if pos_feats_prediction != DEFAULT_OOV_TOKEN:
                    pos_tag, feats_tag = pos_feats_prediction.split('|', 1)
                sentence_pos_tags.append(pos_tag)
                sentence_feats_tags.append(feats_tag)
            pos_tags.append(sentence_pos_tags)
            feats_tags.append(sentence_feats_tags)

        # Restore heads.
        # Heads are integers, so simply convert them to strings.
        heads = [list(map(str, sentence_predictions))
                for sentence_predictions in output["head_predictions"].tolist()]

        # Restore deprels.
        deprels = []
        deprel_ids = output["deprel_prediction_ids"].tolist()
        for sentence_deprel_ids in deprel_ids:
            sentence_deprels = []
            for deprel_id in sentence_deprel_ids:
                deprel = self.vocab.get_token_from_index(deprel_id, "deprel_labels")
                if deprel == DEFAULT_OOV_TOKEN:
                    deprel = '_'
                sentence_deprels.append(deprel)
            deprels.append(sentence_deprels)

        # Restore semslots.
        semslots = self.semslot_classifier.decode_ids(output["semslot_prediction_ids"], oov_token_replacement='_')
        # Restore semclasses.
        semclasses = self.semclass_classifier.decode_ids(output["semclass_prediction_ids"], oov_token_replacement='_')

        return {
            "metadata": sentences,
            "ids": ids,
            "forms": forms,
            "lemmas": lemmas,
            "pos": pos_tags,
            "feats": feats_tags,
            "heads": heads,
            "deprels": deprels,
            "semslots": semslots,
            "semclasses": semclasses,
        }


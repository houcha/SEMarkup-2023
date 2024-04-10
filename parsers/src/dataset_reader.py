from typing import Iterable, List, Dict, Optional

import conllu
import torch

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, TensorField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token as AllenToken

import sys
sys.path.append("..")
from common.token import Token
from common.sentence import Sentence
from common.parse_conllu import parse_conllu_incr

from .lemmatize_helper import predict_lemma_rule
from .multilabel_adjacency_field import MultilabelAdjacencyField


@DatasetReader.register("compreno_ud_dataset_reader")
class ComprenoUDDatasetReader(DatasetReader):
    """
    See https://guide.allennlp.org/reading-data#2 for guidance.
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}):
        super().__init__()
        self.token_indexers = token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as file:
            for sentence in parse_conllu_incr(file):
                yield self.text_to_instance(
                    sentence.tokens,
                    sentence.words,
                    sentence.lemmas,
                    sentence.upos_tags,
                    sentence.xpos_tags,
                    sentence.feats,
                    sentence.heads,
                    sentence.deprels,
                    sentence.deps,
                    sentence.miscs,
                    sentence.semslots,
                    sentence.semclasses,
                    sentence.metadata,
                )

    def text_to_instance(
        self,
        tokens: List[Token],
        words: List[str],
        lemmas: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        feats_tags: List[str] = None,
        heads: List[int] = None,
        deprels: List[str] = None,
        deps: List[Dict[int, List[str]]] = None,
        miscs: List[str] = None,
        semslots: List[str] = None,
        semclasses: List[str] = None,
        metadata: Dict = None,
    ) -> Instance:
        text_field = TextField(list(map(AllenToken, words)), self.token_indexers)

        fields = {}
        fields['words'] = text_field
        fields['sentences'] = MetadataField(tokens)

        if lemmas is not None:
            lemma_rules = [str(predict_lemma_rule(word if word is not None else '', lemma if lemma is not None else ''))
                           for word, lemma in zip(words, lemmas)]
            fields['lemma_rule_labels'] = SequenceLabelField(lemma_rules, text_field, 'lemma_rule_labels')

        if upos_tags is not None and xpos_tags is not None and feats_tags is not None:
            joint_pos_feats = [
                f"{upos_tag}#{xpos_tag}#" + '|'.join([f"{k}={v}" for k, v in feats_tag.items()])
                for upos_tag, xpos_tag, feats_tag in zip(upos_tags, xpos_tags, feats_tags)
            ]
            fields['pos_feats_labels'] = SequenceLabelField(joint_pos_feats, text_field, 'pos_feats_labels')

        if heads is not None and deprels is not None:
            edges: List[Typle[int, int]] = []
            edges_labels: List[str] = []
            for index, (head, relation) in enumerate(zip(heads, deprels)):
                # Skip nulls.
                if head == -1:
                    continue
                assert 0 <= head
                # Hack: start indexing at 0 and replace ROOT with self-loop.
                # It makes parser implementation much easier.
                if head == 0:
                    # Replace ROOT with self-loop.
                    head = index
                else:
                    # If not ROOT, shift token left.
                    head -= 1
                    assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
                edge = (index, head)
                edges.append(edge)
                edges_labels.append(relation)
            fields['deprel_labels'] = MultilabelAdjacencyField(text_field, edges, edges_labels, 'deprel_labels')

        if deps is not None:
            edges: List[Typle[int, int]] = []
            edges_labels: List[str] = []
            for index, token_deps in enumerate(deps):
                assert 0 < len(token_deps), f"Deps must not be empty"
                for head, relations in token_deps.items():
                    assert 0 <= head
                    # Hack: start indexing at 0 and replace ROOT with self-loop.
                    # It makes parser implementation much easier.
                    if head == 0:
                        # Replace ROOT with self-loop.
                        head = index
                    else:
                        # If not ROOT, shift token left.
                        head -= 1
                        assert head != index, f"head = {head + 1} must not be equal to index = {index + 1}"
                    edge = (index, head)
                    edges.append(edge)
                    assert len(relations) == 1, f"Multiedges are not allowed: {deps}, metadata: {metadata}"
                    relation = relations[0]
                    edges_labels.append(relation)
            fields['deps_labels'] = MultilabelAdjacencyField(text_field, edges, edges_labels, 'deps_labels')

        if miscs is not None:
            fields['misc_labels'] = SequenceLabelField(miscs, text_field, 'misc_labels')

        if semslots is not None:
            fields['semslot_labels'] = SequenceLabelField(semslots, text_field, 'semslot_labels')

        if semclasses is not None:
            fields['semclass_labels'] = SequenceLabelField(semclasses, text_field, 'semclass_labels')

        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)

        return Instance(fields)


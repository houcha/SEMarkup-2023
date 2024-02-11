from typing import Iterable, List, Dict, Optional

import conllu
import torch

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, TensorField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token as AllenToken

from .token import Token
from .lemmatize_helper import predict_lemma_rule
from .multilabel_adjacency_field import MultilabelAdjacencyField


class Sentence:
    """
    A wrapper over conllu.models.TokenList.
    """

    def __init__(self, tokens: conllu.models.TokenList):
        self._tokens = [Token(**token) for token in tokens]
        self._metadata = tokens.metadata
        Sentence._renumerate_tokens(self._tokens)

    def _collect_field(self, field_type: str) -> Optional[List]:
        field_values = [getattr(token, field_type) for token in self._tokens]

        # If all fields are None, return None (=no reference labels).
        if all(field is None for field in field_values):
            return None
        return field_values

    @staticmethod
    def _renumerate_tokens(tokens: conllu.models.TokenList):
        """
        Renumerate tokens, so that #NULLs get integer id (e.g. [1, 1.1, 2] turns into [1, 2, 3]).
        This also renumerates 'head' and 'deps' tags accordingly.
        Works inplace.
        """
        old2new_id = {'0': 0} # 0 accounts for ROOT head.

        # Change ids.
        for i, token in enumerate(tokens, 1):
            old_id = token.id
            old2new_id[old_id] = i
            token.id = i

        # Change heads and deps.
        for i, token in enumerate(tokens, 1):
            if token.head is not None:
                token.head = old2new_id[str(token.head)]
            new_deps = {}
            # Special case when deps is empty.
            if token.deps is None:
                continue
            for head, rels in token.deps.items():
                new_deps[old2new_id[head]] = rels
            token.deps = new_deps

    @property
    def tokens(self) -> List[Token]:
        return self._tokens

    @property
    def ids(self) -> List[int]:
        return self._collect_field("ids")

    @property
    def words(self) -> List[str]:
        return self._collect_field("form")

    @property
    def lemmas(self) -> Optional[List[str]]:
        return self._collect_field("lemma")

    @property
    def upos_tags(self) -> Optional[List[str]]:
        return self._collect_field("upos")

    @property
    def xpos_tags(self) -> Optional[List[str]]:
        return self._collect_field("xpos")

    @property
    def feats(self) -> Optional[List[str]]:
        return self._collect_field("feats")

    @property
    def heads(self) -> Optional[List[int]]:
        heads = self._collect_field("head")
        # Replace Nones with -1.
        if heads is not None:
            heads = [-1 if head is None else head for head in heads]
        return heads

    @property
    def deprels(self) -> Optional[List[str]]:
        return self._collect_field("deprel")

    @property
    def deps(self) -> Optional[List[Dict]]:
        return self._collect_field("deps")

    @property
    def miscs(self) -> Optional[List[str]]:
        return self._collect_field("misc")

    @property
    def semslots(self) -> Optional[List[str]]:
        return self._collect_field("semslot")

    @property
    def semclasses(self) -> Optional[List[str]]:
        return self._collect_field("semclass")

    @property
    def metadata(self) -> Dict:
        return self._metadata


# TODO: move to utils
def parse_deps(value: str) -> dict:
    """
    Example:
    >>> parse_deps("26:conj|26:nmod:on|18:advcl:while")
    {'26': ['conj', 'nmod:on'], '18': ['advcl:while']}
    """

    if value == '_':
        return dict()

    deps = {}
    for dep in value.split('|'):
        head, rel = dep.split(':', 1)

        if head not in deps:
            deps[head] = []
        deps[head].append(rel)
    return deps


@DatasetReader.register("compreno_ud_dataset_reader")
class ComprenoUDDatasetReader(DatasetReader):
    """
    See https://guide.allennlp.org/reading-data#2 for guidance.
    """

    def __init__(self, token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}):
        super().__init__()
        self.token_indexers = token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as f:
            texts = f.read()

        sentences = conllu.parse(
            texts,
            fields=["id", "form", "lemma", "upos", "xpos", "feats", "head", "deprel", "deps", "misc", "semslot", "semclass"],
            field_parsers={
                "id": lambda line, i: line[i], # Do not split indexes like 1.1
                "upos": lambda line, i: line[i], # Do not treat _ as None
                "xpos": lambda line, i: line[i], # Do not treat _ as None
                "feats": lambda line, i: line[i],
                "deps": lambda line, i: parse_deps(line[i]),
                "misc": lambda line, i: line[i],
            }
        )

        for sentence in map(Sentence, sentences):
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
        # TODO: exclude NULLs' tags from vocabulary.

        text_field = TextField(list(map(AllenToken, words)), self.token_indexers)

        fields = {}
        fields['words'] = text_field
        fields['sentences'] = MetadataField(tokens)

        if lemmas is not None:
            lemma_rules = [str(predict_lemma_rule(word, lemma)) for word, lemma in zip(words, lemmas)]
            fields['lemma_rule_labels'] = SequenceLabelField(lemma_rules, text_field, 'lemma_rule_labels')

        if upos_tags is not None and xpos_tags is not None and feats_tags is not None:
            joint_pos_feats = [
                f"{upos_tag}#{xpos_tag}#{feats_tag}"
                for upos_tag, xpos_tag, feats_tag in zip(upos_tags, xpos_tags, feats_tags)
            ]
            fields['pos_feats_labels'] = SequenceLabelField(joint_pos_feats, text_field, 'pos_feats_labels')

        if heads is not None and deprels is not None:
            edges = []
            edges_labels = []
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
                edges_labels.append([relation])
            fields['deprel_labels'] = MultilabelAdjacencyField(edges, text_field, edges_labels, 'deprel_labels')

        if deps is not None:
            edges = []
            edges_labels = []
            for index, token_deps in enumerate(deps):
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
                    edges_labels.append(relations)
            fields['deps_labels'] = MultilabelAdjacencyField(edges, text_field, edges_labels, 'deps_labels')

        if miscs is not None:
            fields['misc_labels'] = SequenceLabelField(miscs, text_field, 'misc_labels')

        if semslots is not None:
            fields['semslot_labels'] = SequenceLabelField(semslots, text_field, 'semslot_labels')

        if semclasses is not None:
            fields['semclass_labels'] = SequenceLabelField(semclasses, text_field, 'semclass_labels')

        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)

        return Instance(fields)


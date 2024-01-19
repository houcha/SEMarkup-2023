from typing import Iterable, List, Dict, Optional
from copy import deepcopy

import conllu
import torch

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, TensorField, SequenceLabelField, MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

from .lemmatize_helper import predict_lemma_rule


HEAD_NULL_TOKEN_ID = -1


def is_null(token: Token) -> bool:
    """
    Check whether token is ellipted (null token).
    """
    return token["form"] == "#NULL"


class Sentence:
    """
    A wrapper over conllu.models.TokenList.
    """

    def __init__(self, tokens: conllu.models.TokenList):
        self._tokens = deepcopy(tokens)
        Sentence._renumerate_tokens(self._tokens)
        # Cache mask.
        self._null_mask = [is_null(token) for token in tokens]

    def _collect_field(self, field_type: str) -> Optional[List]:
        field_values = [token[field_type] for token in self._tokens]

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
        #print(tokens.metadata)
        old2new_id = {'0': 0} # 0 to accont for root.

        # Change ids.
        for i, token in enumerate(tokens, 1):
            old_id = token["id"]
            old2new_id[old_id] = i
            token["id"] = i

        #print(f'old2new_id: {old2new_id}')
        # Change heads and deps.
        try:
            for token in tokens:
                if token["head"] is not None:
                    token["head"] = old2new_id[str(token["head"])]
                new_deps = {}
                #print(f'token["deps"]: {token["deps"]}')
                for head, rels in token["deps"].items():
                    new_deps[old2new_id[head]] = rels
                token["deps"] = new_deps
        except Exception as e:
            print(tokens.metadata["sent_id"], e)

    @property
    def null_mask(self) -> List[bool]:
        return self._null_mask

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
        #heads = self._collect_field("head")
        # Replace nulls' heads with HEAD_NULL_TOKEN_ID, since they don't take part in basic UD.
        #return [HEAD_NULL_TOKEN_ID if is_null else head for head, is_null in zip(heads, self.null_mask)]

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
        return self._tokens


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
                sentence.words,
                sentence.null_mask,
                sentence.lemmas,
                sentence.upos_tags,
                sentence.xpos_tags,
                sentence.feats,
                sentence.heads,
                sentence.deprels,
                sentence.semslots,
                sentence.semclasses,
                sentence.metadata,
            )

    def text_to_instance(
        self,
        words: List[str],
        null_mask: List[bool],
        lemmas: List[str] = None,
        upos_tags: List[str] = None,
        xpos_tags: List[str] = None,
        feats_tags: List[str] = None,
        heads: List[int] = None,
        deprels: List[str] = None,
        semslots: List[str] = None,
        semclasses: List[str] = None,
        metadata: Dict = None,
    ) -> Instance:
        # TODO: exclude NULLs' tags from vocabulary.

        text_field = TextField(list(map(Token, words)), self.token_indexers)

        fields = {}
        fields['words'] = text_field
        fields['null_mask'] = TensorField(torch.BoolTensor(null_mask), padding_value=False)

        if lemmas is not None:
            lemma_rules = [str(predict_lemma_rule(word, lemma)) for word, lemma in zip(words, lemmas)]
            fields['lemma_rule_labels'] = SequenceLabelField(lemma_rules, text_field, 'lemma_rule_labels')

        if upos_tags is not None and xpos_tags is not None and feats_tags is not None:
            joint_pos_feats = [
                f"{upos_tag}#{xpos_tag}#{feats_tag}"
                for upos_tag, xpos_tag, feats_tag in zip(upos_tags, xpos_tags, feats_tags)
            ]
            fields['pos_feats_labels'] = SequenceLabelField(joint_pos_feats, text_field, 'pos_feats_labels')

        if heads is not None:
            fields['head_labels'] = SequenceLabelField(heads, text_field, 'head_labels')

        if deprels is not None:
            fields['deprel_labels'] = SequenceLabelField(deprels, text_field, 'deprel_labels')

        if semslots is not None:
            fields['semslot_labels'] = SequenceLabelField(semslots, text_field, 'semslot_labels')

        if semclasses is not None:
            fields['semclass_labels'] = SequenceLabelField(semclasses, text_field, 'semclass_labels')

        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)

        return Instance(fields)


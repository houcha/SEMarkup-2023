from typing import Iterable, List, Dict, Optional

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
        simple_tokens = []
        null_tokens = []
        for token in tokens:
            if is_null(token):
                null_tokens.append(token)
                continue
            simple_tokens.append(token)
        # Move null tokens to the end of the sentence: [<token1>, ..., <tokenN>, <null1>, ..., <nullK>].
        self._tokens = conllu.models.TokenList(simple_tokens + null_tokens, metadata=tokens.metadata)
        # Cache mask.
        self._null_mask = [is_null(token) for token in self._tokens]

    def _collect_field(self, field_type: str) -> Optional[List]:
        field_values = [token[field_type] for token in self._tokens]

        # If all fields are None, return None (=no reference labels).
        if all(field is None for field in field_values):
            return None
        return field_values

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
        # Replace nulls' heads with HEAD_NULL_TOKEN_ID, since they don't take part in basic UD.
        return [HEAD_NULL_TOKEN_ID if is_null else head for head, is_null in zip(heads, self.null_mask)]

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
                "upos": lambda line, i: line[i], # Do not treat _ as None
                "xpos": lambda line, i: line[i], # Do not treat _ as None
                "feats": lambda line, i: line[i],
                "deps": lambda line, i: line[i],
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

        text_field = TextField(list(map(Token, words)), self.token_indexers)

        fields = {}
        fields['words'] = text_field
        fields['null_mask'] = TensorField(torch.BoolTensor(null_mask), padding_value=True)

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


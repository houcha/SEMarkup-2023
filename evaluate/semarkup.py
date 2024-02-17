import conllu

import collections

from copy import deepcopy
from typing import Iterator, Iterable, TextIO, List, Union
from conllu.models import TokenList, Token


SEMARKUP_FIELDS = [
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc",
    "semslot",
    "semclass"
]


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


FIELD_CUSTOM_PARSERS = {
    "id": lambda line, i: line[i],
    "head": lambda line, i: line[i],
    "deps": lambda line, i: parse_deps(line[i]),
    "misc": lambda line, i: line[i]
}


# TODO: merge with parser/scr/token.py and move to common utils
class SemarkupToken:
    def __init__(
        self,
        id = None, # shadow built-in function because ** operator requires exact match with conllu token.
        form = None,
        lemma = None,
        upos = None,
        xpos = None,
        feats = None,
        head = None,
        deprel = None,
        deps = None,
        misc = None,
        semslot = None,
        semclass = None
    ):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats if feats is not None else {}
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc
        self.semslot = semslot
        self.semclass = semclass

    def is_null(self) -> bool:
        """
        Check whether token is ellipted (null token).
        """
        return self.form == "#NULL"

    def is_empty(self) -> bool:
        """
        Check whether token is empty token.
        """
        return self.form == "#EMPTY"

    @staticmethod
    def create_null(id: str):
        return Token(id, "#NULL")


# Special tokens.
EMPTY_TOKEN = SemarkupToken(form="#EMPTY")
CLS_TOKEN = SemarkupToken(id=0, form="[CLS]")


class Sentence:
    def __init__(self, sentence: TokenList):
        self.sentence = sentence
        self.sent_id = sentence.metadata['sent_id']

    def __getitem__(self, index: int) -> SemarkupToken:
        return SemarkupToken(**self.sentence[index])

    def __len__(self) -> int:
        return len(self.sentence)

    def serialize(self) -> str:
        sentence_copy = deepcopy(self.sentence)
        # Manually serialize deps tag.
        for token in sentence_copy:
            if not token["deps"]:
                # If deps is empty, assign _.
                token["deps"] = ''
                continue

            dep_str_list = []
            for head, rels in token["deps"].items():
                for rel in rels:
                    dep_str_list.append(f"{head}:{rel}")
            token["deps"] = '|'.join(dep_str_list)
        return sentence_copy.serialize()


class SentenceIterator(collections.abc.Iterator):
    def __init__(self, conllu_sentences: Iterable[TokenList]):
        self.conllu_sentences = conllu_sentences

    def __next__(self) -> Sentence:
        return Sentence(next(self.conllu_sentences))

def parse_semarkup(file: TextIO, incr: bool) -> Union[SentenceIterator, List[TokenList]]:
    assert not file.closed

    if incr:
        # Return SentenceIterator
        sentences = SentenceIterator(conllu.parse_incr(file, fields=SEMARKUP_FIELDS, field_parsers=FIELD_CUSTOM_PARSERS))
    else:
        # Return list
        sentences = conllu.parse(file.read(), fields=SEMARKUP_FIELDS, field_parsers=FIELD_CUSTOM_PARSERS)

    return sentences


def write_semarkup(file_path: str, sentences: List[TokenList]) -> None:
    sentences_serialized = []
    for sentence in sentences:
        sentence_filtered = deepcopy(sentence)
        for token in sentence_filtered:
            for extra_tag in set(token.keys()) - set(SEMARKUP_FIELDS):
                token.pop(extra_tag)
        sentences_serialized.append(Sentence(sentence_filtered).serialize())
    with open(file_path, 'w') as file:
        file.write(''.join(sentences_serialized))


import conllu

from copy import deepcopy
from typing import Iterable, TextIO, List, Optional, Dict

from .sentence import Sentence


FIELDS = [
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

def parse_nullable_value(value: str) -> Optional[str]:
    return value if value else None

def parse_feats(value: str) -> Optional[Dict[str, str]]:
    if not value:
        return None

    if value == '_':
        return dict()

    return {
        part.split("=")[0]: part.split("=")[1] if "=" in part else ""
        for part in value.split("|") if part.split("=")[0]
    }

def parse_head(value: str) -> Optional[int]:
    if not value:
        return None
    elif value == '_':
        return -1
    return int(value)

def parse_deps(value: str) -> Optional[Dict[str, List[str]]]:
    """
    Example:
    >>> parse_deps("26:conj|18:advcl:while")
    {'26': 'conj', '18': 'advcl:while'}
    """

    if not value:
        return None

    if value == '_':
        return dict()

    deps = {}
    for dep in value.split('|'):
        head, rel = dep.split(':', 1)
        if head not in deps:
            deps[head] = []
        deps[head].append(rel)
    return deps


FIELD_PARSERS = {
    "id": lambda line, i: line[i], # Do not split indexes like 1.1
    "lemma": lambda line, i: parse_nullable_value(line[i]),
    "upos": lambda line, i: parse_nullable_value(line[i]), # Do not treat _ as None
    "xpos": lambda line, i: parse_nullable_value(line[i]), # Do not treat _ as None
    "feats": lambda line, i: parse_feats(line[i]),
    "head": lambda line, i: parse_head(line[i]),
    "deps": lambda line, i: parse_deps(line[i]),
    "misc": lambda line, i: parse_nullable_value(line[i])
}

def parse_conllu_raw(file: TextIO) -> List[conllu.models.TokenList]:
    return conllu.parse(file.read(), fields=FIELDS, field_parsers=FIELD_PARSERS)

def parse_conllu(file: TextIO) -> List[Sentence]:
    return list(map(Sentence.from_conllu, parse_conllu_raw(file)))

def parse_conllu_incr(file: TextIO) -> Iterable[Sentence]:
    assert not file.closed
    return map(Sentence.from_conllu, conllu.parse_incr(file, fields=FIELDS, field_parsers=FIELD_PARSERS))

def write_conllu(file_path: str, sentences: List[Sentence]) -> None:
    sentences_serialized: List[str] = []

    for sentence in sentences:
        sentences_serialized.append(sentence.serialize())

    with open(file_path, 'w') as file:
        file.write(''.join(sentences_serialized))


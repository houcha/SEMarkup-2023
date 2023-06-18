import conllu

import collections

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
    "semslot",
    "semclass"
]


class SemarkupToken:
    def __init__(self, conllu_token: Token):
        self.id = conllu_token["id"]
        self.form = conllu_token["form"]
        self.lemma = conllu_token["lemma"]
        self.upos = conllu_token["upos"]
        self.pos = conllu_token["upos"] # ALIAS
        self.xpos = conllu_token["xpos"]
        self.feats = conllu_token["feats"] if conllu_token["feats"] is not None else {}
        self.head = conllu_token["head"]
        self.deprel = conllu_token["deprel"]
        self.semslot = conllu_token["semslot"]
        self.semclass = conllu_token["semclass"]


class Sentence:
    def __init__(self, sentence: TokenList):
        self.sentence = sentence
        self.sent_id = sentence.metadata['sent_id']

    def __getitem__(self, index: int) -> SemarkupToken:
        return SemarkupToken(self.sentence[index])

    def __len__(self) -> int:
        return len(self.sentence)

    def serialize(self) -> str:
        return self.sentence.serialize()


class SentenceIterator(collections.abc.Iterator):
    def __init__(self, conllu_sentences: Iterable[TokenList]):
        self.conllu_sentences = conllu_sentences

    def __next__(self) -> Sentence:
        return Sentence(next(self.conllu_sentences))


def parse_semarkup(file: TextIO, incr: bool) -> Union[SentenceIterator, List[SemarkupToken]]:
    assert not file.closed

    if incr:
        # Return SentenceIterator
        sentences = SentenceIterator(conllu.parse_incr(file, fields=SEMARKUP_FIELDS))
    else:
        # Return list
        sentences = conllu.parse(file.read(), fields=SEMARKUP_FIELDS)

    return sentences


def write_semarkup(file_path: str, sentences: List[TokenList]) -> None:
    sentences_serialized = []
    for sentence in sentences:
        sentence_filtered = TokenList(
            [Token({field: token[field] for field in SEMARKUP_FIELDS}) for token in sentence]
        )
        sentences_serialized.append(sentence_filtered.serialize())
    with open(file_path, 'w') as file:
        file.write(''.join(sentences_serialized))


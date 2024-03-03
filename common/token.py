from typing import Dict, Union

from conllu.serializer import serialize_field


class Token:
    def __init__(
        self,
        id: Union[int, str] = None, # shadow built-in function because ** operator requires exact match with conllu token.
        form: str = None,
        lemma: str = None,
        upos: str = None,
        xpos: str = None,
        feats: Dict[str, str] = None,
        head: Union[int, str] = None,
        deprel: str = None,
        deps: Dict[str, str] = None,
        misc: str = None,
        semslot: str = None,
        semclass: str = None
    ):
        self.id = str(id)
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = int(head) if head is not None else None
        self.deprel = deprel
        self.deps = deps
        self.misc = misc
        self.semslot = semslot
        self.semclass = semclass

    def serialize(self) -> str:
        # Custom serialization for 'head' tag.
        head = None
        if self.head is None:
            head = ''
        elif self.head == -1:
            head = '_'
        else:
            head = str(head)

        # Custom serialization for 'deps' tag.
        deps = ''
        if self.deps:
            dep_str_list = []
            for head, rels in self.deps.items():
                for rel in rels:
                    dep_str_list.append(f"{head}:{rel}")
            deps = '|'.join(dep_str_list)
        return (
            f"{serialize_field(self.id)}\t"
            f"{serialize_field(self.form)}\t"
            f"{serialize_field(self.lemma)}\t"
            f"{serialize_field(self.upos)}\t"
            f"{serialize_field(self.xpos)}\t"
            f"{serialize_field(self.feats)}\t"
            f"{head}\t"
            f"{serialize_field(self.deprel)}\t"
            f"{deps}\t"
            f"{serialize_field(self.misc)}\t"
            f"{serialize_field(self.semslot)}\t"
            f"{serialize_field(self.semclass)}"
        )

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

    ### Special tokens.

    @staticmethod
    def create_null(id: str):
        return Token(id, "#NULL")

    @staticmethod
    def create_empty(id: str):
        return Token(id, "#EMPTY")


CLS_TOKEN = Token(id=0, form="[CLS]")


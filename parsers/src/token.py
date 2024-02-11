class Token:
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
        self.feats = feats
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
        return self.form is None

    @staticmethod
    def create_null(id: str):
        return Token(id, "#NULL")


# Special tokens.
EMPTY_TOKEN = Token(form="#EMPTY")
CLS_TOKEN = Token(id=0, form="[CLS]")

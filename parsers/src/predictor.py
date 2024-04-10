from overrides import override
from typing import Dict

from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance

import sys
sys.path.append("..")
from common.token import Token
from common.sentence import Sentence


@Predictor.register("morpho_syntax_semantic_predictor")
class MorphoSyntaxSemanticPredictor(Predictor):
    """
    See https://guide.allennlp.org/training-and-prediction#4 for guidance.
    """

    @override(check_signature=False)
    def dump_line(self, output: Dict[str, list]) -> str:
        metadata = output["metadata"]

        lines = []
        if metadata:
            for key, value in metadata.items():
                if value:
                    line = f"# {key} = {value}"
                else:
                    line = f"# {key}"
                lines.append(line)

        tags_iterator = zip(
            output["ids"],
            output["forms"],
            output["lemmas"],
            output["upos"],
            output["xpos"],
            output["feats"],
            output["heads"],
            output["deprels"],
            output["deps"],
            output["miscs"],
            output["semslots"],
            output["semclasses"],
        )
        lines.extend(['\t'.join(map(str, token_tags)) for token_tags in tags_iterator])
        return '\n'.join(lines) + "\n\n"


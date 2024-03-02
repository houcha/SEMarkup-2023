#!/usr/bin/env python3

import sys
import json
import argparse
from copy import deepcopy
from conllu.models import TokenList, Token

from tqdm import tqdm

from typing import Dict, Iterable, List

sys.path.append('../../evaluate')
from semarkup import parse_semarkup, write_semarkup, Sentence


def preprocess(sentences: List[TokenList]) -> List[str]:
    serialized_sentences = []
    for sentence in tqdm(sentences):
        tokens = []
        for token in sentence:
            if '-' in token["id"]:
                continue
            deps = deepcopy(token["deps"])
            token["deps"] = dict()
            for head, rels in deps.items():
                try:
                    a, b = head.split('-')
                    head = a
                except:
                    pass
                token["deps"][head] = rels
            tokens.append(token)

        preprocessed_sentence = Sentence(TokenList(tokens, sentence.metadata))
        serialized_sentence = preprocessed_sentence.serialize()
        serialized_sentences.append(serialized_sentence)
    return serialized_sentences


def main(input_file_path: str, output_file_path: str) -> None:
    print(f"Load sentences...")
    with open(input_file_path, "r", encoding='utf8') as input_file:
        sentences = parse_semarkup(input_file, incr=False)

    print("Processing...")
    preprocessed_sentences = preprocess(sentences)

    print("Writing results")
    with open(output_file_path, 'w') as output_file:
        output_file.write(''.join(preprocessed_sentences))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess SEMarkup file (e.g. remove "x-y" indices), so that parser can work with it.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='SEMarkup file to preprocess.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Preprocessed file.'
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)


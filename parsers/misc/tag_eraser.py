#!/usr/bin/env python3

import sys
import argparse
from tqdm import tqdm

sys.path.append("../..") # import 'common' package
from common.parse_conllu import parse_conllu, write_conllu
from common.sentence import Sentence


def main(
    input_file_path: str,
    output_file_path: str,
    keep_syntax: bool
) -> None:

    with open(input_file_path, "r") as file:
        sentences = parse_conllu(file)

    for sentence in tqdm(sentences):
        for token in sentence:
            token.lemma = ""
            token.upos = ""
            token.xpos = ""
            token.feats = ""

            if not keep_syntax:
                token.head = ""
                token.deprel = ""
                token.deps = ""

            token.misc = ""
            token.semslot = ""
            token.semclass = ""
    write_conllu(output_file_path, sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Remove tags from conllu file, leaving `id` and `form` intact.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input conllu file.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output conllu file with tags removed.'
    )
    parser.add_argument(
        '--keep-syntax',
        action='store_true',
        help='If set, syntax tags (head, deprel and deps) are preserved.'
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file, args.keep_syntax)


#!/usr/bin/env python3

import sys
import argparse
from copy import deepcopy

from tqdm import tqdm

from typing import Dict, Iterable, List
from conllu.models import TokenList

sys.path.append('../..')
from common.parse_conllu import parse_conllu_raw, write_conllu
from common.sentence import Sentence, Token


def preprocess(token_lists: List[TokenList]) -> List[Sentence]:
    processed_sentences = []
    for token_list in tqdm(token_lists):
        tokens = []
        for token in token_list:
            # Skip range tokens.
            if '-' in token["id"]:
                continue
            tokens.append(Token(**token))
        processed_sentences.append(Sentence(tokens, token_list.metadata))
    return processed_sentences


def main(input_file_path: str, output_file_path: str) -> None:
    print(f"Load sentences...")
    with open(input_file_path, "r", encoding='utf8') as input_file:
        # Use parse_conllu_raw, because it preserves tags as-is (e.g. doesn't renumerate tokens' ids)
        token_lists = parse_conllu_raw(input_file)
    print("Processing...")
    preprocessed_sentences = preprocess(token_lists)
    print("Writing results")
    write_conllu(output_file_path, preprocessed_sentences)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Preprocess SEMarkup file (e.g. remove "x-y" indices), so that parser can work with it.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Conllu file to preprocess.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Preprocessed file.'
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)


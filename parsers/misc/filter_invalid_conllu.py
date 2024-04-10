#!/usr/bin/env python3

import sys
import argparse
from typing import Dict, Iterable, List
from tqdm import tqdm

sys.path.append("../..") # import 'common' package
from common.parse_conllu import parse_conllu_incr, write_conllu
from common.sentence import Sentence


def is_int(s: str):
    try:
        int(s)
    except ValueError:
        return False
    except TypeError:
        return False
    else:
        return True


def validate_conllu(sentences: Iterable[Sentence], verbose: bool = True) -> List[Sentence]:
    valid_sentences = []
    total_sentences_count = 0

    for sentence in tqdm(sentences):
        total_sentences_count += 1

        try:
            sentence_all_ids = set()
            sentence_natural_ids = set() # without #NULL
            sentence_heads = set()
            sentence_deps_heads = set()
            roots_count = 0
            
            for token in sentence:
                sentence_all_ids.add(int(token.id))
                if is_int(token.id):
                    sentence_natural_ids.add(int(token.id))

                # Null validation
                if token.is_null():
                    assert token.form == "#NULL", f"Did you mean #NULL?: {token.form}"
                    assert token.head == -1, f"Null's head = {token.head} != _"
                    assert token.deprel == '_', f"Null's deprel = {token.deprel} != _"
                    assert len(token.deps) > 0, f"Null's deps = {token.deps} != _"
                    assert token.misc == 'ellipsis', f"Null's misc != ellipsis"
                    continue

                # Head
                # Non-null tokens must have non-empty head.
                assert is_int(token.head), f"Non-null tokens must have integer head. Encountered: {token.head} at {token.id}"
                token.head = int(token.head)
                assert token.head != int(token.id), f"Self-loops are not allowed in heads. Head: {token.head}"
                assert 0 <= token.head, f"Head must be non-negative. Encountered {token.head}"
                if token.head == 0:
                    roots_count += 1
                sentence_heads.add(token.head)

                # Deps
                token_heads = set()
                assert 1 <= len(token.deps), f"{token.form} has empty deps: {token.deps}"
                for head, rels in token.deps.items():
                    assert len(rels) == 1, \
                        f"Multiedges are not allowed: {rels}"
                    assert is_int(head) or head is None, \
                        f"Deps head must be either int or null (x.1). Encountered: {head}"
                    assert str(head) != str(token.id), \
                        f"Self-loops are not allowed in deps. Head: {head}"
                    assert 0 <= float(head), \
                        f"Deps head must be non-negative. Encountered: {head}"
                    token_heads.add(head)
                    sentence_deps_heads.add(head)

            ids_diff = set(range(min(sentence_natural_ids), max(sentence_natural_ids))) - sentence_natural_ids
            assert not ids_diff, f"Sentence ids are non-continuous, absent ids: {ids_diff}"
            head_ids_diff = sentence_heads - sentence_natural_ids - {0}
            assert not head_ids_diff, f"Heads' ids are not consistent with ids, extra ids: {head_ids_diff}"
            deps_head_ids_diff = sentence_deps_heads - sentence_all_ids - {0}
            assert not deps_head_ids_diff, f"Deps heads' ids are not consistent with ids, extra ids: {deps_head_ids_diff}"\
                f"sentence_all_ids: {sentence_all_ids}, sentence_deps_heads: {sentence_deps_heads}"
            assert roots_count == 1, "There must be one ROOT (head=0) in a sentence."

            valid_sentences.append(sentence)

        except AssertionError as e:
            if verbose:
                print(f"Error: {e}")
                print(f"Sentence:\n{sentence.serialize()}")

    valid_sentences_fraction = len(valid_sentences) / total_sentences_count
    print(f"Number of valid sentences: {len(valid_sentences)}, which is {valid_sentences_fraction:2f} of total dataset size.")

    return valid_sentences


def main(input_file_path: str, output_file_path: str):
    print(f"Load sentences...")
    with open(input_file_path, "r", encoding='utf8') as file:
        sentences = parse_conllu_incr(file)
        valid_sentences = validate_conllu(sentences, verbose=True)
        write_conllu(output_file_path, valid_sentences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conllu format validity check.')
    parser.add_argument(
        'input_file',
        type=str,
        help='Conllu file to validate.'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Output file with valid sentences only.'
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)


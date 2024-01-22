import sys
import json
import argparse

from tqdm import tqdm

from typing import Dict, Iterable

sys.path.insert(0,'..')
from semarkup import parse_semarkup, Sentence


def load_dict_from_json(json_file_path: str) -> Dict:
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data


def is_int(s: str):
    try:
        int(s)
    except ValueError:
        return False
    except TypeError:
        return False
    else:
        return True

def is_null(s: str):
    try:
        x, y = s.split('.')
        if y != '1':
            return False
    except ValueError:
        return False
    else:
        return True


def validate_semarkup(sentences: Iterable[Sentence], vocab_file: str = None) -> bool:
    if vocab_file is not None:
        vocab = load_dict_from_json(vocab_file)
        vocab["upos"] = set(vocab["upos"])
        vocab["xpos"] = set(vocab["xpos"])
        for cat, grams in vocab["feats"].items():
            vocab["feats"][cat] = set(grams)
        vocab["deprels"] = set(vocab["deprels"])
        vocab["semslots"] = set(vocab["semslots"])
        vocab["semclasses"] = set(vocab["semclasses"])

    is_valid = True

    for sentence in tqdm(sentences):
        try:
            sentence_all_ids = set()
            sentence_natural_ids = set() # without #NULL
            sentence_heads = set()
            sentence_deps_heads = set()
            roots_count = 0
            
            for token in sentence:
                sentence_all_ids.add(token.id)
                if is_int(token.id):
                    sentence_natural_ids.add(int(token.id))

                # Null validation
                if "#null" in token.form.lower():
                    assert token.form == "#NULL", f"Did you mean #NULL?: {token.form}"
                    assert token.head == '_', f"Null's head = {token.head} != _"
                    token.head = None
                    assert token.deprel == '_', f"Null's deprel = {token.deprel} != _"
                    assert len(token.deps) > 0, f"Null's deps = {token.deps} != _"
                    assert token.misc == 'ellipsis', f"Null's misc != ellipsis"
                    continue
                
                # UPOS
                if vocab_file is not None:
                    assert token.upos in vocab["upos"], \
                        f"UPOS {token.upos} is out of vocabulary."

                # XPOS
                if vocab_file is not None:
                    assert token.xpos is None or token.xpos in vocab["xpos"], \
                        f"XPOS {token.xpos} is out of vocabulary."

                # Feats
                if vocab_file is not None:
                    for cat, gram in token.feats.items():
                        assert cat in vocab["feats"], \
                            f"grammatical category {cat} is out of vocabulary."
                        assert gram in vocab["feats"][cat], \
                            f"grammeme {gram} is out of vocabulary."

                # Head
                # Non-null tokens must have non-empty head.
                assert is_int(token.head), f"Non-null tokens must have integer head."
                token.head = int(token.head)
                if is_int(token.id):
                    assert token.head != int(token.id), f"Self-loops are not allowed in heads. Head: {token.head}"
                assert 0 <= token.head, f"Head must be non-negative. Encountered {token.head}"
                if token.head == 0:
                    roots_count += 1
                sentence_heads.add(token.head)

                # Deps
                for head, rels in token.deps.items():
                    assert is_int(head) or is_null(head), \
                        f"Deps head must be either int or null (x.1). Encountered: {head}"
                    assert head != token.id, \
                        f"Self-loops are not allowed in deps. Head: {head}"
                    assert 0 <= float(head), \
                        f"Deps head must be non-negative. Encountered: {head}"
                    sentence_deps_heads.add(head)

                # Semslot
                if vocab_file is not None:
                    assert token.semslot in vocab["semslots"], \
                        f"Semslot {token.semslot} is out of vocabulary."

                # Semclass
                if vocab_file is not None:
                    assert token.semclass in vocab["semclasses"], \
                        f"Semclass {token.semclass} is out of vocabulary."

            ids_diff = set(range(min(sentence_natural_ids), max(sentence_natural_ids))) - sentence_natural_ids
            assert not ids_diff, f"Sentence ids are non-continuous, absent ids: {ids_diff}"
            head_ids_diff = sentence_heads - sentence_natural_ids - {0}
            assert not head_ids_diff, f"Heads' ids are not consistent with ids, extra ids: {head_ids_diff}"
            deps_head_ids_diff = sentence_deps_heads - sentence_all_ids - {'0'}
            assert not deps_head_ids_diff, f"Deps heads' ids are not consistent with ids, extra ids: {deps_head_ids_diff}"
            assert roots_count == 1, "There must be one ROOT (head=0) in a sentence."

        except AssertionError as e:
            print(f"Error: {e}")
            print(f"Sentence:\n{sentence.serialize()}")
            is_valid = False

    return is_valid


def main(semarkup_file_path: str, vocab_file: str) -> None:
    print(f"Load sentences...")
    with open(semarkup_file_path, "r", encoding='utf8') as semarkup_file:
        sentences = parse_semarkup(semarkup_file, incr=True)
        if validate_semarkup(sentences, vocab_file):
            print("Seems legit!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SEMarkup sanity check.')
    parser.add_argument(
        'semarkup_file',
        type=str,
        help='SEMarkup file to check.'
    )
    parser.add_argument(
        '-vocab_file',
        type=str,
        help="JSON file with ground-truth vocabulary (use build_vocab.py to build one)."
        "For example, you can use it if you have a correct train SEMarkup file and "
        "want to make sure test SEMarkup file doesn't have tags which are not present in test (OOV).",
        default=None
    )
    args = parser.parse_args()
    main(args.semarkup_file, args.vocab_file)


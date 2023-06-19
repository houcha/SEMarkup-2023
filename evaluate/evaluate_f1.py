import argparse

import numpy as np
from sklearn.metrics import f1_score

from semarkup import parse_semarkup


def map_str_to_int(str_array: list) -> list:
    str_to_int = dict((key, i) for i, key in enumerate(set(str_array)))
    int_array = list(map(str_to_int.get, str_array))
    return int_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SEMarkup-2023 evaluation script.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'test_file',
        type=str,
        help='Test file in SEMarkup format with predicted tags.'
    )
    parser.add_argument(
        'gold_file',
        type=str,
        help="Gold file in SEMarkup format with true tags.\n"
        "For example, SEMarkup-2023-Evaluate/train.conllu."
    )
    args = parser.parse_args()

    with open(args.test_file, 'r') as test_file, open(args.gold_file, 'r') as gold_file:
        test_sentences = parse_semarkup(test_file, incr=False)
        gold_sentences = parse_semarkup(gold_file, incr=False)

    tagsets_names = ["upos", "feats", "semslot", "semclass"]

    test_tags = {tagset_name: [] for tagset_name in tagsets_names}
    gold_tags = {tagset_name: [] for tagset_name in tagsets_names}

    assert len(test_sentences) == len(gold_sentences)
    for test_sentence, gold_sentence in zip(test_sentences, gold_sentences):
        assert len(test_sentence) == len(gold_sentence)
        for test_token, gold_token in zip(test_sentence, gold_sentence):
            for tagset_name in tagsets_names:
                # Apply str() to tags, because some tags (feats) are of unhashable types
                test_tags[tagset_name].append(str(test_token[tagset_name]))
                gold_tags[tagset_name].append(str(gold_token[tagset_name]))

    print(f"         macro  micro")
    for tagset_name in tagsets_names:
        tags_concat = map_str_to_int(test_tags[tagset_name] + gold_tags[tagset_name])
        test_tags_int = tags_concat[:len(test_tags[tagset_name])]
        gold_tags_int = tags_concat[len(test_tags[tagset_name]):]
        assert len(test_tags_int) == len(test_tags[tagset_name])
        assert len(gold_tags_int) == len(gold_tags[tagset_name])

        macro_f1 = f1_score(gold_tags_int, test_tags_int, average='macro')
        micro_f1 = f1_score(gold_tags_int, test_tags_int, average='micro')

        print(f"{tagset_name:8} {macro_f1*100:.1f}%  {micro_f1*100:.1f}%")


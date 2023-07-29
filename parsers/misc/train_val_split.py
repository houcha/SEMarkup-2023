#!/usr/bin/env python3

import sys
import argparse
from random import shuffle

from conllu.models import TokenList

sys.path.append('../src')
from lemmatize_helper import predict_lemma_rule

sys.path.append('../../evaluate')
from semarkup import parse_semarkup, write_semarkup


def calc_tagsets_tags_sizes(
    sentences: list[TokenList],
    tagsets_names: list[str]
) -> dict[str, dict[str, int]]:

    tagsets_tags_sizes = dict()
    for tagset_name in tagsets_names:
        tags_sizes = dict()
        for sentence in sentences:
            for token in sentence:
                tag = token[tagset_name]
                if tag not in tags_sizes:
                    tags_sizes[tag] = 0
                tags_sizes[tag] += 1
        tagsets_tags_sizes[tagset_name] = tags_sizes
    return tagsets_tags_sizes


def calc_desirable_train_tagsets_tags_sizes(
    tagsets_tags_sizes: dict[str, dict[str, int]],
    train_fraction: int
) -> dict[str, dict[str, int]]:

    desirable_train_tagsets_tags_sizes = dict()
    for tagset_name, tags_sizes in tagsets_tags_sizes.items():
        desirable_tags_sizes = dict()
        for tag, tag_size in tags_sizes.items():
            desirable_tags_sizes[tag] = int(train_fraction * tag_size) if tag_size > 1 else 1
        desirable_train_tagsets_tags_sizes[tagset_name] = desirable_tags_sizes
    return desirable_train_tagsets_tags_sizes


def build_inverted_tagsets(
    entences: list[TokenList],
    tagsets_names: list[str]
) -> dict[str, dict[str, set[int]]:
    sentences_tagsets = []
    tagsets = {tagset_name: set() for tagset_name in tagsets_names}

    for sentence in sentences:
        sentence_tagsets = dict()
        for tagset_name in tagsets_names:
            sentence_tagsets[tagset_name] = {str(token[tagset_name]) for token in sentence}
            tagsets[tagset_name] |= sentence_tagsets[tagset_name]
        sentences_tagsets.append(sentence_tagsets)

    # inv_tagsets is a dict of inverted tagsets.
    # Inverted tagset is a dict of (tag -> indexes of sentences containing this tag).
    inv_tagsets = {tagset_name: {tag: set() for tag in tagsets[tagset_name]} for tagset_name in tagsets_names}
    for sentence_index, sentence_tagsets in enumerate(sentences_tagsets):
        for tagset_name in tagsets_names:
            for tag in sentence_tagsets[tagset_name]:
                inv_tagsets[tagset_name][tag].add(sentence_index)
    return inv_tagsets


def find_most_rare_tag(
    tagsets_tags_sizes: dict[str, dict[str, int]],
    total_tagsets_tags_sizes: dict[str, dict[str, int]]
) -> tuple[str, str]:

    rarest_tag, rarest_tag_tagset_name = None, None
    rarest_tag_count, rerest_tag_total_count = float('inf'), float('inf')
    for tagset_name, tags_sizes in tagsets_tags_sizes.items():
        tags_sizes, total_tags_sizes = tagsets_tags_sizes[tagset_name], total_tagsets_tags_sizes[tagset_name]
        tags_priorities = {tag: (tags_sizes[tag], total_tags_sizes[tag]) for tag in tags_sizes}
        rare_tag, (rare_tag_count, rare_tag_total_count) = min(tags_priorities.items(), key=lambda x: x[1])
        if rare_tag_count < rarest_tag_count or (rare_tag_count == rarest_tag_count and rare_tag_total_count < rarest_tag_total_count):
            rarest_tag = rare_tag
            rarest_tag_tagset_name = tagset_name
            rarest_tag_count = rare_tag_count
            rarest_tag_total_count = rare_tag_total_count
    return rarest_tag, rarest_tag_tagset_name


def subtract_sentences_from_tags_sizes(
    tagsets_tags_sizes: dict[str, dict[str, int]],
    sentences: list[TokenList]
) -> dict[str, set[str]]:

    # Search for saturated tags.
    tagsets_tags_saturated = {tagset_name: set() for tagset_name in tagsets_tags_sizes}
    for tagset_name, tags_sizes in tagsets_tags_sizes.items():
        tags_saturated = tagsets_tags_saturated[tagset_name]
        for sentence in sentences:
            for token in sentence:
                tag = token[tagset_name]
                tags_sizes[tag] -= 1
                if tags_sizes[tag] == 0:
                    tags_saturated.add(tag)
    # Remove saturated tags from statistic.
    for tagset_name, tags_saturated in tagsets_tags_saturated.items():
        tagset_tags_sizes = tagsets_tags_sizes[tagset_name]
        for tag_saturated in tags_saturated:
            tagset_tags_sizes.pop(tag_saturated)
            if not tagset_tags_sizes:
                tagsets_tags_sizes.pop(tagset_name)
    return tagsets_tags_saturated


def subtract_sentences_from_inverted_tagsets(
    inv_tagsets: dict[str, dict[str, set[int]]],
    sentences_indexes: set[int]
) -> None:

    for inv_tagset in inv_tagsets.values():
        inv_tagset_keys = list(inv_tagset.keys()) # Make a copy so that inverted tagset can be updated in-place.
        for tag in inv_tagset_keys:
            sentences_with_tag = inv_tagset[tag]
            sentences_with_tag -= sentences_indexes
            # If there are no sentences left for a tag, remove it.
            if not sentences_with_tag:
                inv_tagset.pop(tag)


def build_train_dataset(
    sentences: list[TokenList],
    tagsets_names: list[str],
    train_fraction: float
) -> set[int]:
    """
    Build a set of sentences such that every tag from `tagsets` tagsets has at least one occurrence.
    We use this to make sure training set contains all tags of all tagsets.
    """

    inv_tagsets = build_inverted_tagsets(sentences, tagsets_names)
    tags_sizes = calc_tagsets_tags_sizes(sentences, tagsets_names)
    desirable_train_tags_sizes = calc_desirable_train_tagsets_tags_sizes(tags_sizes, train_fraction)

    train_sentences_indexes = set()
    # While desirable set is not collected.
    while desirable_train_tags_sizes:
        # Find the rarest tag over all tagsets.
        rarest_tag, rarest_tag_tagset_name = find_most_rare_tag(desirable_train_tags_sizes, tags_sizes)
        assert rarest_tag is not None

        try:
            # Since tags are gathered greedily, the sentences with a rarest tag might have been excluded from the index
            # on the previous steps in order not to gather saturated tags.
            sentences_with_tag = inv_tagsets[rarest_tag_tagset_name][rarest_tag]
        except:
            # If that is the case, i.e. if current rarest tag was among excluded sentences, then just ignore it.
            # The resulting size of the tag is going to be smaller than the desirable one, but usually it's not a big problem.
            assert rarest_tag not in inv_tagsets[rarest_tag_tagset_name]
            drop_count = desirable_train_tags_sizes[rarest_tag_tagset_name].pop(rarest_tag)
            # Cleanup tagset if no tags left in it.
            if not desirable_train_tags_sizes[rarest_tag_tagset_name]:
                desirable_train_tags_sizes.pop(rarest_tag_tagset_name)
            continue
        # Pick random sentence with this tag.
        sentence_with_tag = list(sentences_with_tag)[0]

        # Add sentence to training set.
        train_sentences_indexes.add(sentence_with_tag)

        # Delete sentence from inverted tagsets.
        subtract_sentences_from_inverted_tagsets(inv_tagsets, {sentence_with_tag})
        # Update desirable tags sizes.
        tagsets_tags_saturated = subtract_sentences_from_tags_sizes(desirable_train_tags_sizes, [sentences[sentence_with_tag]])

        sentences_with_saturated_tags = set()
        # Find all sentences containing saturated tags...
        for tagset_name, tags_saturated in tagsets_tags_saturated.items():
            for tag_saturated in tags_saturated:
                inv_tagset = inv_tagsets[tagset_name]
                # Tag might have been deleted at the previous subtract_sentences_from_inverted_tagsets call.
                if tag_saturated in inv_tagset:
                    sentences_with_saturated_tag = inv_tagset[tag_saturated]
                    sentences_with_saturated_tags |= sentences_with_saturated_tag
        # ...and delete them, as we don't want to add sentences with saturated tags to training set anymore.
        subtract_sentences_from_inverted_tagsets(inv_tagsets, sentences_with_saturated_tags)

    return train_sentences_indexes


def train_val_split(
    sentences: list[TokenList],
    tagsets_names: list[str],
    train_fraction: float
) -> None:
    assert 0.0 < train_fraction < 1.0, "train_fraction must be in (0, 1) range."

    train_sentence_indexes = build_train_dataset(sentences, tagsets_names, train_fraction)
    validation_sentence_indexes = set(range(len(sentences))) - train_sentence_indexes

    train_sentences = [sentences[i] for i in train_sentence_indexes]
    validation_sentences = [sentences[i] for i in validation_sentence_indexes]

    return train_sentences, validation_sentences


def print_dataset_statistic(sentences: list[TokenList], tagsets_names: list[str]) -> None:
    print(f"Number of sentences: {len(sentences)}")
    for tagset_name in tagsets_names:
        tagset_size = len({token[tagset_name] for sentence in sentences for token in sentence})
        print(f"{tagset_name} tagset size: {tagset_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Split dataset file into train and validation files.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'dataset',
        type=str,
        help='Dataset file in SEMarkup format to be splitted.'
    )
    parser.add_argument(
        'train_file',
        type=str,
        help='A train file to be produced.'
    )
    parser.add_argument(
        'validation_file',
        type=str,
        help='A validation file to be produced.'
    )
    parser.add_argument(
        'train_fraction',
        type=float,
        help='A fraction of train part. A real number smaller than 1.'
    )
    args = parser.parse_args()

    print("Loading sentences...")
    with open(args.dataset, 'r') as file:
        sentences = parse_semarkup(file, incr=False)

    # Modify tags.
    for sentence in sentences:
        for token in sentence:
            token["upos&feats"] = token["upos"] + "&" + str(token["feats"])

    tagsets_names = ["semclass", "semslot", "upos&feats"]

    print("Splitting...")
    train_sentences, validation_sentences = train_val_split(sentences, tagsets_names, args.train_fraction)

    print()
    print("==============================")
    print("All sentences statistic")
    print("------------------------------")
    print_dataset_statistic(train_sentences + validation_sentences, tagsets_names)
    print("==============================")
    print()
    print("==============================")
    print("Train sentences statistic")
    print("------------------------------")
    print_dataset_statistic(train_sentences, tagsets_names)
    print("==============================")
    print()
    print("==============================")
    print("Validation sentences statistic")
    print("------------------------------")
    print_dataset_statistic(validation_sentences, tagsets_names)
    print("==============================")
    print()

    write_semarkup(args.train_file, train_sentences)
    write_semarkup(args.validation_file, validation_sentences)


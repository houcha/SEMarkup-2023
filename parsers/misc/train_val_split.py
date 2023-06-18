#!/usr/bin/env python3

import sys
import argparse
from random import shuffle

from typing import List, Set
from conllu.models import TokenList

sys.path.append('../src')
from lemmatize_helper import predict_lemma_rule

sys.path.append('../../evaluate')
from semarkup import parse_semarkup, write_semarkup


def build_cover_sentences(sentences: List[TokenList], tagsets_names: List[str]) -> Set[int]:
    """
    Build a set of sentences such that every tag from `tagsets` tagsets has at least one occurrence.
    We use this to make sure training set contains all the tags of all tagsets.
    """

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

    cover_sentences_indexes = set()

    # While there is non-empty inverted tagset, i.e. we have "uncovered" tags, do the following:
    while sum(len(inv_tagset) for inv_tagset in inv_tagsets.values()) > 0:
        # Find the rarest tag over all tagsets.
        rarest_tag, rarest_tag_frequency, rarest_tag_tagset_name = None, float('inf'), None
        for tagset_name, inv_tagset in inv_tagsets.items():
            if len(inv_tagset) == 0:
                continue
            rare_tag, rare_tag_sentences = min(inv_tagset.items(), key=lambda item: len(item[1]))
            # Number of sentences with this tag.
            rare_tag_frequency = len(rare_tag_sentences)
            if rare_tag_frequency < rarest_tag_frequency:
                rarest_tag = rare_tag
                rarest_tag_frequency = rare_tag_frequency
                rarest_tag_tagset_name = tagset_name
        assert rarest_tag is not None

        # Remove tag from tagset.
        sentences_with_tag = inv_tagsets[rarest_tag_tagset_name].pop(rarest_tag)
        # Pick any sentence with this tag.
        sentence_with_tag = list(sentences_with_tag)[0]
        # Add sentence to training set
        cover_sentences_indexes.add(sentence_with_tag)
        # ...and also remove all tags present in this sentence from all inverted indexes.
        for tagset_name, inv_tagset in inv_tagsets.items():
            for tag in list(inv_tagset.keys()):
                sentences = inv_tagset[tag]
                if sentence_with_tag in sentences:
                    inv_tagset.pop(tag)

    return cover_sentences_indexes


def train_val_split( sentences: List[TokenList], train_fraction: float, tagsets_names: List[str]) -> None:
    assert 0.0 < train_fraction < 1.0, "train_fraction must be in (0, 1) range."
    train_size = int(train_fraction * len(sentences))

    sentences_count = len(sentences)
    train_fraction = 0.8
    train_size = int(train_fraction * sentences_count)
    val_size = sentences_count - train_size

    min_train_sentences_indexes = build_cover_sentences(sentences, tagsets_names)
    assert len(min_train_sentences_indexes) <= train_size
    train_sentences = [sentences[i] for i in min_train_sentences_indexes]
    # Remove sentences that have been added to the initial train set.
    sentences = [sentence for i, sentence in enumerate(sentences) if i not in min_train_sentences_indexes]
    train_size_updated = train_size - len(train_sentences)

    min_val_sentences_indexes = build_cover_sentences(sentences, tagsets_names)
    assert len(min_val_sentences_indexes) <= val_size
    val_sentences = [sentences[i] for i in min_val_sentences_indexes]
    # Remove sentences that have been added to the initial validation set.
    sentences = [sentence for i, sentence in enumerate(sentences) if i not in min_val_sentences_indexes]

    shuffle(sentences)

    train_sentences += sentences[:train_size_updated]
    val_sentences += sentences[train_size_updated:]

    assert len(train_sentences) == train_size
    assert len(train_sentences) + len(val_sentences) == sentences_count
    assert abs((len(train_sentences) / sentences_count) - train_fraction) <= 1e-3
    # Make sure train and validation sets have no common sentences.
    train_sentences_ids = set(int(sentence.metadata['sent_id']) for sentence in train_sentences)
    val_sentences_ids = set(int(sentence.metadata['sent_id']) for sentence in val_sentences)
    assert not (train_sentences_ids & val_sentences_ids)

    return train_sentences, val_sentences


def print_dataset_statistic(sentences: List[TokenList], tagsets_names: List[str]):
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
        'val_file',
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
            token["lemma_rule"] = predict_lemma_rule(token["form"], token["lemma"])
            token["upos&feats"] = token["upos"] + "&" + str(token["feats"])

    tagsets_names = ["lemma_rule", "upos&feats", "semslot", "semclass"]

    print("Splitting...")
    train_sentences, val_sentences = train_val_split(sentences, args.train_fraction, tagsets_names)

    print()
    print("==============================")
    print("All sentences statistic")
    print("------------------------------")
    print_dataset_statistic(train_sentences + val_sentences, tagsets_names)
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
    print_dataset_statistic(val_sentences, tagsets_names)
    print("==============================")
    print()

    write_semarkup(args.train_file, train_sentences)
    write_semarkup(args.val_file, val_sentences)


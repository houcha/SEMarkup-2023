from typing import Iterable, List, Dict, Optional

import torch
from torch import BoolTensor

from .token import Token, EMPTY_TOKEN


def get_null_mask(sentences: List[List[Token]]) -> BoolTensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.BoolTensor([token.is_null() for token in sentence]) for sentence in sentences],
        batch_first=True,
        padding_value=False
    )


def align_tokens(true_tokens: List[Token], pred_tokens: List[Token]):
    """
    Aligns two sequences of tokens. EMPTY_TOKEN is inserted where needed.
    Example:
    >>> true_tokens = [ "How", "did", "this", "#NULL", "happen"]
    >>> pred_tokens = [ "How", "#NULL", "did", "this", "happen"]
    >>> align_labels(true_tokens, pred_tokens)
    ['How', '#EMPTY', 'did', 'this',  '#NULL', 'happen'],
    ['How',  '#NULL', 'did', 'this', '#EMPTY', 'happen']
    """
    true_tokens_aligned = []
    pred_tokens_aligned = []

    i, j = 0, 0
    while i < len(true_tokens) and j < len(pred_tokens):
        if true_tokens[i].form == pred_tokens[j].form:
            true_tokens_aligned.append(true_tokens[i])
            pred_tokens_aligned.append(pred_tokens[j])
            i += 1
            j += 1
        else:
            if true_tokens[i].is_null():
                true_tokens_aligned.append(true_tokens[i])
                pred_tokens_aligned.append(EMPTY_TOKEN)
                i += 1
            else:
                assert pred_tokens[j].is_null()
                true_tokens_aligned.append(EMPTY_TOKEN)
                pred_tokens_aligned.append(pred_tokens[j])
                j += 1
    return true_tokens_aligned, pred_tokens_aligned

def align_batch_of_tokens(true_tokens_batch: List[List[Token]], pred_tokens_batch: List[List[Token]]):
    """
    A wrapper over align_tokens for batched inputs.
    """
    true_tokens_aligned_batch, pred_tokens_aligned_batch  = [], []

    assert len(true_tokens_batch) == len(pred_tokens_batch)
    for true_tokens, pred_tokens in zip(true_tokens_batch, pred_tokens_batch):
        true_tokens_aligned, pred_tokens_aligned = align_tokens(true_tokens, pred_tokens)
        assert len(true_tokens_aligned) == len(pred_tokens_aligned)
        true_tokens_aligned_batch.append(true_tokens_aligned)
        pred_tokens_aligned_batch.append(pred_tokens_aligned)
    return true_tokens_aligned_batch, pred_tokens_aligned_batch


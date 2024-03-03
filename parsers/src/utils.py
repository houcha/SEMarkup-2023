from typing import Iterable, List, Dict, Optional

import torch
from torch import BoolTensor

import sys
sys.path.append("..")
from common.token import Token


def get_null_mask(sentences: List[List[Token]]) -> BoolTensor:
    return torch.nn.utils.rnn.pad_sequence(
        [torch.BoolTensor([token.is_null() for token in sentence]) for sentence in sentences],
        batch_first=True,
        padding_value=False
    )


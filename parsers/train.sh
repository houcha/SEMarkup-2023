#!/usr/bin/env bash

# Build target vocabulary.
allennlp build-vocab configs/build_target_vocab.jsonnet serialization/target_vocab.tar.gz \
    --include-package src \
    --force

# Pretrain model on external datasets.
allennlp train configs/baseline_pretrain.jsonnet \
    --serialization-dir serialization/pretrained \
    --include-package src \
    --force 

# Fine-tune pretrained model on target dataset.
allennlp train configs/baseline_finetune.jsonnet \
    --serialization-dir serialization/finetuned \
    --include-package src \
    --force

#!/usr/bin/env bash

# Build target vocabulary.
allennlp build-vocab $1/build_target_vocab.jsonnet $2/target_vocab.tar.gz \
    --include-package src \
    --force

# Pretrain model on external datasets.
allennlp train $1/pretrain.jsonnet \
    --serialization-dir $2/pretrained \
    --include-package src \
    --override '{
        "vocabulary.directory": "'$2'/target_vocab.tar.gz"
    }' \
    --force

# Fine-tune pretrained model on target dataset.
allennlp train $1/finetune.jsonnet \
    --serialization-dir $2/finetuned \
    --include-package src \
    --override '{
        "vocabulary.directory": "'$2'/pretrained/vocabulary",
        "model.archive_file": "'$2'/pretrained/model.tar.gz"
    }' \
    --force

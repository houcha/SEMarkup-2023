#!/usr/bin/env bash

mkdir ../data
./preprocessing.py ../data/dataset.conllu ../data/dataset_processed.conllu
./filter_invalid_conllu.py ../data/dataset_processed.conllu ../data/dataset_processed_filtered.conllu
./train_val_split.py ../data/dataset_processed_filtered.conllu ../data/train.conllu ../data/validation.conllu 0.8

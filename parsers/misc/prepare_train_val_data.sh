#!/usr/bin/env bash

mkdir ../data
./preprocessing.py ../data/dataset.conllu ../data/dataset_processed.conllu
./train_val_split.py ../data/dataset_processed.conllu ../data/train.conllu ../data/val.conllu 0.8

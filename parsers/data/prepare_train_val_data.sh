#!/usr/bin/env bash

./preprocessing.py dataset.conllu dataset_processed.conllu
./filter_invalid_conllu.py dataset_processed.conllu dataset_processed_filtered.conllu
./train_val_split.py dataset_processed_filtered.conllu train.conllu validation.conllu 0.8

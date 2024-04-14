#!/usr/bin/env bash
#
# Download English treebanks with enhanced UD markup.

# https://github.com/UniversalDependencies/UD_English-GUM
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-dev.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/master/en_gum-ud-test.conllu
cat en_gum-ud-train.conllu en_gum-ud-dev.conllu en_gum-ud-test.conllu > en_gum.conllu
rm en_gum-ud-train.conllu en_gum-ud-dev.conllu en_gum-ud-test.conllu

# https://github.com/UniversalDependencies/UD_English-EWT
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-dev.conllu
wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-test.conllu
cat en_ewt-ud-train.conllu en_ewt-ud-dev.conllu en_ewt-ud-test.conllu > en_ewt.conllu
rm en_ewt-ud-train.conllu en_ewt-ud-dev.conllu en_ewt-ud-test.conllu

# Merge different datasets into one.
cat en_gum.conllu en_ewt.conllu > en_pretrain.conllu
rm en_gum.conllu en_ewt.conllu

# Remove extra tokens.
./preprocessing.py en_pretrain.conllu en_pretrain.conllu
# Remove all tags but syntactic ones, since we want to pretrain on syntax only.
./tag_eraser.py en_pretrain.conllu en_pretrain.conllu --keep-syntax
# Remove ill-formatted sentences.
./filter_invalid_conllu.py en_pretrain.conllu en_pretrain.conllu

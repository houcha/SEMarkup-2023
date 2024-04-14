#!/usr/bin/env bash

git clone https://github.com/CobaldAnnotation/CobaldEng
mv CobaldEng/train.conllu dataset.conllu
rm -rf CobaldEng

#!/usr/bin/env bash

CONFIG_FILE=$1
SERIALIZATION_DIR=$2
PRETRAINING_DATASET=data/en_pretrain.conllu
FINETUNING_DATASET=data/train.conllu

# We have two stages: pretraining and finetuning.
# In order to be able to pretrain model on one dataset and finetune it on another one,
# we must build common vocabulary before pretraining (because clasification heads size depends on
# vocabulary size). The easiest (yet not the most effective) way to do this in allennlp is to
# merge both pretraining and target conllu files into one.
MERGED_DATASET=data/merged_dataset.conllu
cat $PRETRAINING_DATASET $FINETUNING_DATASET > $MERGED_DATASET

# Build common vocabulary.
COMMON_VOCAB_FILE=$SERIALIZATION_DIR/common_vocab.tar.gz
allennlp build-vocab $CONFIG_FILE $COMMON_VOCAB_FILE \
    --overrides '{
        "train_data_path": "'$MERGED_DATASET'"
    }' \
    --include-package src
# We don't need common conllu anymore.
rm $MERGED_DATASET

# Pretrain model on external datasets.
allennlp train $CONFIG_FILE \
    --serialization-dir $SERIALIZATION_DIR/pretrained \
    --override '{
        "train_data_path": "'$PRETRAINING_DATASET'",
        "vocabulary": {
            "type": "from_files",
            "directory": "'$COMMON_VOCAB_FILE'",
        },
    }' \
    --include-package src
PRETRAINED_MODEL_PATH=$SERIALIZATION_DIR/pretrained/model.tar.gz

# Fine-tune pretrained model on target dataset.
allennlp train $CONFIG_FILE \
    --serialization-dir $SERIALIZATION_DIR/finetuned \
    --override '{
        "train_data_path": "'$FINETUNING_DATASET'",
        "vocabulary": {
            "type": "from_files",
            "directory": "'$COMMON_VOCAB_FILE'",
        },
        "model": {
            "type": "from_archive",
            "archive_file": "'$PRETRAINED_MODEL_PATH'"
        },
    }' \
    --include-package src

# Now manually build model.tar.gz, so that it's not bound to the local directories.
TMP_SERIALIZATION_DIR=$SERIALIZATION_DIR/tmp
mkdir $TMP_SERIALIZATION_DIR
# model.tar.gz requires weights (best.th), vocabulary and config.json.
# 1. Copy best weights from finetuned model.
cp $SERIALIZATION_DIR/finetuned/best.th $TMP_SERIALIZATION_DIR
# 2. Copy vocabulary.
cp -r $SERIALIZATION_DIR/finetuned/vocabulary $TMP_SERIALIZATION_DIR
# 3. Create config.json from CONFIG_FILE.
python3 -c '
import _jsonnet
print(_jsonnet.evaluate_file("'$CONFIG_FILE'"))
' > $TMP_SERIALIZATION_DIR/config.json
# Build model.tar.gz from tmp directory.
python3 -c '
from allennlp.models.archival import archive_model
archive_model("'$TMP_SERIALIZATION_DIR'")'
# Move model at the top level.
mv $TMP_SERIALIZATION_DIR/model.tar.gz $SERIALIZATION_DIR
# Remove temporary directory.
rm -rf $TMP_SERIALIZATION_DIR

echo "Done."


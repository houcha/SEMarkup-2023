{
    "train_data_path": "data/train.conllu",
    "validation_data_path": "data/validation.conllu",
    "dataset_reader": {
        "type": "compreno_ud_dataset_reader", # Use custom dataset reader.
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "cointegrated/rubert-tiny" # Use rubert-tiny.
            }
        },
    },
    "vocabulary": {
        "min_count": {
            "lemma_rule_labels": 2 # Ignore lemmatization rules encountered 1 time in training dataset.
        },
        "tokens_to_add": { # Add default OOV tokens.
            "lemma_rule_labels": ["@@UNKNOWN@@"],
        }
    },
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    }
}

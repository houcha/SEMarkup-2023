{
    "train_data_path": "data/train_small_no_multiedges.conllu",
    "validation_data_path": "data/validation_small_no_multiedges.conllu",
    "dataset_reader": {
        "type": "compreno_ud_dataset_reader", # Use custom dataset reader.
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
		# WARNING: don't forget to change model.embedder and model.indexer as well.
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

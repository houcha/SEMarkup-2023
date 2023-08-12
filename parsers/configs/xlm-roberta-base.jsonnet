# Configuration file for parser.
# See https://guide.allennlp.org/training-and-prediction#2 for guidance.
{
    "train_data_path": "data/train.conllu",
    "validation_data_path": "data/val.conllu",
    "dataset_reader": {
        "type": "compreno_ud_dataset_reader", # Use custom dataset reader.
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "xlm-roberta-base"
            }
        },
    },
    "data_loader": {
        "batch_size": 32, # Batch of this size takes up to 13Gb of GPU memory.
        "shuffle": true
    },
    "validation_data_loader": {
        "batch_size": 32,
        "shuffle": false
    },
    "vocabulary": {
        "type": "vocabulary_weighted",
        "min_count": {
            "lemma_rule_labels": 2 # Ignore lemmatization rules encountered 1 time in training dataset.
        },
        "tokens_to_add": { # Add default OOV tokens.
            "lemma_rule_labels": ["@@UNKNOWN@@"],
        }
    },
    "model": {
        "type": "morpho_syntax_semantic_parser", # Use custom model.
        "embedder": {
            "type": "pretrained_transformer_mismatched",
            "model_name": "xlm-roberta-base", # Use pretrained encoder.
            "train_parameters": true
        },
        "lemma_rule_classifier": {
            "hid_dim": 512,
            "activation": "relu",
            "dropout": 0.1,
            "paradigm_dictionary_path": "dicts/MorphoDic_full_utf8_abbrs.txt",
            "dictionary_lemmas_info": [
                {
                    "path": "dicts/Compreno.txt",
                    "lemma_match_pattern": "\\d+:(.*)"
                },
                {
                    "path": "dicts/Zaliz.txt",
                    "lemma_match_pattern": "^(.*?) "
                },
                { # TODO: There should be a task-independent dictionary with pronouns.
                    "path": "data/train.conllu",
                    "lemma_match_pattern": "^\\d+\\s+.*?\\s+(.*?)\\s+"
                },
            ],
            "topk": 10,
        },
        "pos_feats_classifier": {
            "hid_dim": 1024,
            "activation": "relu",
            "dropout": 0.1,
        },
        "dependency_classifier": {
            "hid_dim": 128,
            "activation": "relu",
            "dropout": 0.1
        },
        "semslot_classifier": {
            "hid_dim": 1024,
            "activation": "relu",
            "dropout": 0.1,
        },
        "semclass_classifier": {
            "hid_dim": 1024,
            "activation": "relu",
            "dropout": 0.1,
        },
        "initializer": {
            "regexes": [
                ["lemma_rule_classifier.*", "normal"],
                ["pos_feats_classifier.*", "normal"],
                ["dependency_classifier.*", "normal"],
                ["semslot_classifier.*", "normal"],
                ["semclass_classifier.*", "normal"]
            ]
        }
    },
    "trainer": {
        "type": "gradient_descent",
        "optimizer": {
            "type": "adam",
            "lr": 0.01, # Base (largest) learning rate.
            "parameter_groups": [
                [ # Second group of layers.
                    ["embedder"], {}
                ],
                [ # First group of layers.
                    [
                    "lemma_rule_classifier",
                    "pos_feats_classifier",
                    "dependency_classifier",
                    "semslot_classifier",
                    "semclass_classifier"
                    ], {}
                ],
            ],
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "gradual_unfreezing": true, # During first epoch the first group of layers is trained only. Starting second epoch, both groups are trained.
            "discriminative_fine_tuning": true, # Enable discriminative finetuning.
            "decay_factor": 0.005, # We want RoBERTa to be trained with learning rate 200 times smaller than heads.
            "cut_frac": 0.0, # Increase learning rate from the smallest to the base value instantly.
            "ratio": 32, # The ratio of the smallest to the largest (base) learning rate.
        },
        "callbacks": [
            { # Enable Tensorboard logs. Can we viewed via "tensorboard --logdir serialization_dir".
                "type": "tensorboard",
                "should_log_parameter_statistics": false,
                "should_log_learning_rate": true,
            }
        ],
        "num_epochs": 15,
        "validation_metric": "+AverageAccuracy", # Track average score of all scores. '+' stands for 'higher - better'.
        "grad_clipping": 5.0, # Clip gradient if too high.
        "cuda_device": 1, # GPU
    }
}

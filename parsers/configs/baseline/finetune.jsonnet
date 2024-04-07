# Configuration file for baseline model finetuning (it must be pretrained using baseline_pretrain.jsonnet config first!).
{
    "train_data_path": "data/train.conllu",
    "validation_data_path": "data/validation.conllu",
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
    "data_loader": {
        "batch_size": 16,
        "shuffle": true
    },
    "validation_data_loader": {
        "batch_size": 16,
        "shuffle": false
    },
    "vocabulary": {
        "type": "from_files",
        "directory": "serialization/pretrained/vocabulary",
    },
    "model": {
	"type": "from_archive",
	"archive_file": "serialization/pretrained/model.tar.gz"
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
                    "misc_classifier",
                    "semslot_classifier",
                    "semclass_classifier",
                    "null_classifier"
                    ], {}
                ],
            ],
        },
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "gradual_unfreezing": true, # During first epoch the first group of layers is trained only. Starting second epoch, both groups are trained.
            "discriminative_fine_tuning": true, # Enable discriminative finetuning.
            "decay_factor": 0.01, # We want BERT to be trained with learning rate 100 times smaller than heads.
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
        "num_epochs": 10,
        "validation_metric": "+Avg", # Track average score of all scores. '+' stands for 'higher - better'.
        "grad_clipping": 5.0, # Clip gradient if too high.
        "cuda_device": 0, # GPU
    }
}

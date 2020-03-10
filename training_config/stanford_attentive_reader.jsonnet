{
  "dataset_reader": {
    "type": "cnn_dm"
  },
  "train_data_path": "datasets/cnn_train.jsonl",
  "validation_data_path": "datasets/cnn_dev.jsonl",
  "model": {
    "type": "stanford_attentive_reader",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz",
          "embedding_dim": 100,
          "trainable": true
        },
      }
    },
    "question_encoder": {
        "type": "gru",
        "input_size": 100,
        "hidden_size": 128,
        "num_layers": 1,
        "bidirectional": true
    },
    "context_encoder": {
        "type": "lstm",
        "input_size": 100,
        "hidden_size": 128,
        "num_layers": 1,
        "bidirectional": true
    },
    "matrix_attention": {
      "type": "bilinear",
      "matrix_1_dim": 128,
      "matrix_2_dim": 128
    }
  },
  "data_loader": {
    "batch_sampler": {
    "type": "bucket",
    "padding_noise": 0.0,
    "batch_size": 32
    },
  },
  "trainer": {
    "num_epochs": 30,
    "grad_norm": 10.0,
    "patience" : 10,
    "cuda_device" : 0,
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}

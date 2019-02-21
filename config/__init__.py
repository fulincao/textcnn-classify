class TextCnnConfig(object):

    model_save_dir = './checkpoints'
    model_file_prefix = 'text-cnn-model'
    embedding_dimension = 64
    sequence_length = 200
    num_classes = 30
    num_filters = 128
    kernel_size = 5
    vocab_size = 5000
    hidden_dimension = 128
    dropout_keep_prob = 0.5
    learning_rate = 1e-3
    batch_size = 64
    num_epochs = 2
    save_per_batch = 10
    train_device = "/cpu:0"

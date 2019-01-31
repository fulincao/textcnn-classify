import tensorflow as tf


class TextCNN(object):

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):

        with tf.device(self.config.train_device):
            my_init = tf.glorot_uniform_initializer()
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dimension],
                                    initializer= my_init)
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        with tf.name_scope("cnn"):
            convs = []
            for i in range(-3, 3):
                conv0 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size + i, kernel_initializer=my_init)
                conv0 = tf.layers.batch_normalization(conv0)
                conv0 = tf.nn.relu(conv0)
                pool0 = tf.reduce_max(conv0, reduction_indices=[1], name='gmp')
                convs.append(pool0)
            max_pool = tf.concat(convs, 1)
            print(max_pool.shape)
            print(max_pool.shape)

        with tf.name_scope("score"):
            fc = tf.layers.dense(max_pool, self.config.hidden_dimension, name='fc1',kernel_initializer=my_init)

            fc = tf.layers.batch_normalization(fc)
            fc = tf.nn.relu(fc)
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)

            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.predict_label = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope("optimizer"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            correct_predict = tf.equal(tf.argmax(self.input_y, 1), self.predict_label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

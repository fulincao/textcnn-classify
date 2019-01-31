import os
import json
import argparse
import tensorflow as tf
import tensorflow.contrib.keras as keras
from config import TextCnnConfig
from models.cnn_model import TextCNN
from utils import preprocess as UTILS

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s')


class CnnModel:
    def __init__(self, config):
        self.config = config
        self.model = TextCNN(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=os.path.join(config.model_save_dir, config.model_file_prefix))


    def predict(self, content):

        data = [WORD_2_ID[word] for word in content if word in WORD_2_ID]
        logging.debug("")

        feed_dict = {
            self.model.input_x: keras.preprocessing.sequence.pad_sequences([data], self.config.sequence_length),
            self.model.keep_prob: 1.0
        }

        predict_label = self.session.run(self.model.predict_label, feed_dict=feed_dict)
        return LABELS[predict_label[0]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./dataset",
                        help="The Base Directory of Dataset")
    parser.add_argument("--data_dir", type=str, default="asks",
                        help="The Base Directory of Asks of 169 kang")
    parser.add_argument("--labels_json_file", type=str, default="labels.json",
                        help="json file of labels")

    parser.add_argument("--predict", type=str, default=u"脚好痛")

    # global variables here
    FLAGS, unknown = parser.parse_known_args()
    TEST_FILE = os.path.join(FLAGS.base_dir, FLAGS.data_dir, 'test.new.txt')
    # VALID_FILE = os.path.join(FLAGS.base_dir, FLAGS.data_dir, 'val.txt')
    VOCAB_FILE = os.path.join(FLAGS.base_dir, FLAGS.data_dir, 'vocab.txt')

    LABELS = json.loads(open(os.path.join(FLAGS.base_dir, FLAGS.data_dir, FLAGS.labels_json_file), encoding='utf-8').read())
    LABEL_2_ID = UTILS.generates_label_map(LABELS)
    WORD_2_ID = UTILS.generates_vocab_map(VOCAB_FILE)

    TEXT_CNN_CONFIG = TextCnnConfig()
    TEXT_CNN_CONFIG.vocab_size = UTILS.obtain_vocab_size(VOCAB_FILE)
    cnn_model = CnnModel(TEXT_CNN_CONFIG)

    predict_label = cnn_model.predict(FLAGS.predict)
    logging.debug("category is {}".format(predict_label))

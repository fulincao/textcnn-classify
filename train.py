import sys
import os
import json
import argparse

import numpy as np
import tensorflow as tf
from sklearn import metrics

from config import TextCnnConfig
from models.cnn_model import TextCNN
from utils import preprocess as UTILS

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s: %(message)s')



def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch = UTILS.generates_batch(x_, y_, 128)
    total_loss = 0.0
    total_accuracy = 0.0

    for x_batch, y_batch in batch:
        batch_len = len(x_batch)
        feed_dict = {
            MODEL.input_x: x_batch,
            MODEL.input_y: y_batch,
            MODEL.keep_prob: 1.0
        }
        loss, accuracy = sess.run([MODEL.loss, MODEL.accuracy], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_accuracy += accuracy * batch_len

    return total_loss / data_len, total_accuracy / data_len


def test(word_2_id, label_2_id):
    logging.debug("Loading test data...")
    x_test, y_test = UTILS.obtain_inputs_of_cnn(TEST_FILE, word_2_id, label_2_id, TEXT_CNN_CONFIG.sequence_length)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=os.path.join(TEXT_CNN_CONFIG.model_save_dir,
                                                       TEXT_CNN_CONFIG.model_file_prefix))

    logging.debug('Testing...')
    test_loss, test_accuracy = evaluate(session, x_test, y_test)
    info = 'Test Loss: {0:>6.2}, Test Accuracy: {1:>7.2%}'
    logging.debug(info.format(test_loss, test_accuracy))

    data_len = len(x_test)
    num_batch = int((data_len - 1) / TEXT_CNN_CONFIG.batch_size) + 1

    correct_label = np.argmax(y_test, 1)
    predict_label = np.zeros(shape=len(x_test), dtype=np.int32)
    batch_size = 128

    for index in range(num_batch):
        start = index * batch_size
        end = min((index + 1) * batch_size, data_len)
        feed_dict = {
            MODEL.input_x: x_test[start:end],
            MODEL.keep_prob: 1.0
        }
        predict_label[start:end] = session.run(MODEL.predict_label, feed_dict=feed_dict)


    logging.debug("Precision, Recall and F1-Score...")
    logging.debug(metrics.classification_report(correct_label, predict_label, target_names=LABELS))

    logging.debug("Confusion Matrix...")
    cm = metrics.confusion_matrix(correct_label, predict_label)
    logging.debug(cm)


def main(unused_argv):
    '''
    :param unused_argv: unused
    :return:void
    '''
    _ = unused_argv

    saver = tf.train.Saver()
    if not os.path.exists(TEXT_CNN_CONFIG.model_save_dir):
        logging.debug("Creating checkpoints directory:{}".format(TEXT_CNN_CONFIG.model_save_dir))
        os.makedirs(TEXT_CNN_CONFIG.model_save_dir)

    logging.debug("Loading training and validation data...")

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    label_2_id = UTILS.generates_label_map(LABELS)
    UTILS.generates_vocab_file(TRAIN_FILE, VOCAB_FILE, label_2_id, TEXT_CNN_CONFIG.vocab_size)
    word_2_id = UTILS.generates_vocab_map(VOCAB_FILE)

    logging.debug("Generating training and validation inputs...")
    x_train, y_train = UTILS.obtain_inputs_of_cnn(TRAIN_FILE, word_2_id, label_2_id, TEXT_CNN_CONFIG.sequence_length)
    x_valid, y_valid = UTILS.obtain_inputs_of_cnn(VALID_FILE, word_2_id, label_2_id, TEXT_CNN_CONFIG.sequence_length)


    logging.debug('Training and evaluating...')

    total_batch = 0
    best_valid_accuracy = 0.0
    last_improved = 0
    requirements = 10000
    has_been_optimized = False

    for epoch in range(TEXT_CNN_CONFIG.num_epochs):
        logging.debug('Epoch:{0}'.format(epoch + 1))
        batch = UTILS.generates_batch(x_train, y_train, TEXT_CNN_CONFIG.batch_size)

        for x_batch, y_batch in batch:

            feed_dict = {
                MODEL.input_x: x_batch,
                MODEL.input_y: y_batch,
                MODEL.keep_prob: TEXT_CNN_CONFIG.dropout_keep_prob
            }

            if total_batch % FLAGS.print_interval == 0:

                feed_dict[MODEL.keep_prob] = 1.0
                train_loss, train_accuracy = session.run([MODEL.loss, MODEL.accuracy], feed_dict=feed_dict)
                valid_loss, valid_accuracy = evaluate(session, x_valid, y_valid)

                if valid_accuracy > best_valid_accuracy:
                    best_valid_accuracy = valid_accuracy
                    last_improved = total_batch
                    saver.save(sess=session, save_path=os.path.join(TEXT_CNN_CONFIG.model_save_dir,
                                                                    TEXT_CNN_CONFIG.model_file_prefix))

                info = 'Iter: {0:>6}, Train Loss: {1:>6.4}, Train Accuracy: {2:>7.4%},' \
                      + ' Valid Loss: {3:>6.4}, Valid Accuracy: {4:>7.4%}'
                logging.debug(info.format(total_batch, train_loss, train_accuracy, valid_loss, valid_accuracy))

            session.run(MODEL.optimizer, feed_dict=feed_dict)
            total_batch += 1

            if total_batch - last_improved > requirements:
                logging.debug("Model has come to be perfect, auto-stopping...")
                has_been_optimized = True
                break

        if has_been_optimized:
            break

    logging.debug("Testing the trained model:")
    test(word_2_id, label_2_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./dataset",
                        help="The Base Directory of Dataset")
    parser.add_argument("--data_dir", type=str, default="asks",
                        help="The Base Directory of Asks of 169 kang")
    parser.add_argument("--labels_json_file", type=str, default="labels.json",
                        help="json file of labels")
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--train_device", type=str, default="cpu")

    # global variables here
    FLAGS, unknown = parser.parse_known_args()
    TRAIN_FILE = os.path.join(FLAGS.base_dir, FLAGS.data_dir, 'train.txt')
    TEST_FILE = os.path.join(FLAGS.base_dir, FLAGS.data_dir, 'test.txt')
    VALID_FILE = os.path.join(FLAGS.base_dir, FLAGS.data_dir, 'val.txt')
    VOCAB_FILE = os.path.join(FLAGS.base_dir, FLAGS.data_dir, 'vocab.txt')
    LABELS = json.loads(open(os.path.join(FLAGS.base_dir, FLAGS.data_dir, FLAGS.labels_json_file), encoding='utf-8').read())
    TEXT_CNN_CONFIG = TextCnnConfig()
    TEXT_CNN_CONFIG.train_device = "/{}:0".format(FLAGS.train_device)
    MODEL = TextCNN(TEXT_CNN_CONFIG)

    tf.app.run(main=main, argv=[sys.argv[0]] + unknown)

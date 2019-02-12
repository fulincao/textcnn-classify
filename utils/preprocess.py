from collections import Counter
import numpy as np
import tensorflow.contrib.keras as keras


def _parse_train_file(filename, label_2_id, separate=':'):
    '''
    :param filename: each line's format like the following:label$content\n
    :return:
    '''

    labels, contents = [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                content = line.strip().split(separate)
                label = content[0]
                content = ''.join(content[1:])
                if not label or label not in label_2_id:
                    continue
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except Exception as e:
                print(e)
                pass

    return labels, contents


def generates_vocab_file(train_file, vocab_file, label_2_id, vocab_size=5000):
    '''

    :param train_file:each line's format like the following:label$content\n
    :param vocab_file: the file to write the most common vocab
    :param vocab_size: default 5000
    :return:
    '''

    _, contents = _parse_train_file(train_file, label_2_id)

    all_words = []
    for content in contents:
        all_words.extend(content)

    counter = Counter(all_words)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)
    with open(vocab_file, "w", encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')


def generates_vocab_map(vocab_file):
    '''
    :param vocab_file:
    :return:
    '''

    with open(vocab_file, "r", encoding='utf-8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def obtain_vocab_size(vocab_file):
    '''
    :param vocab_file:
    :return:
    '''
    with open(vocab_file, "r", encoding='utf-8') as fp:
        words = [_.strip() for _ in fp.readlines()]
    return len(words)

def generates_label_map(labels):
    '''
    :param config: obtain labels from config file:config/__init__.py
    :return:{label:ID,...}
    '''
    return dict(zip(labels, range(len(labels))))



def obtain_inputs_of_cnn(train_file, word_to_id, label_to_id, max_sequence_length=200):
    '''
    :param train_file:
    :param word_to_id:{word:index,...}
    :param label_to_id:{label:index,...}
    :param max_length:
    :return:
    '''

    labels, contents = _parse_train_file(train_file, label_to_id)
    single_word_indices, label_indices = [], []

    for index in range(len(contents)):
        single_word_indices.append([word_to_id[word] for word in contents[index] if word in word_to_id])
        label_indices.append(label_to_id[labels[index]])

            
    '''
    single_word_indices=[
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666],
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666],
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666],
    ...
    [1,3,66,7,8,99,213,4,6,7,999,1234,...666]
    ]
    
    label_indices=[1,1,1,2,2,2,3,3,3...128]
    '''
    train_inputs = keras.preprocessing.sequence.pad_sequences(single_word_indices, max_sequence_length)
    train_inputs = np.concatenate([train_inputs, train_inputs])
    label_inputs = keras.utils.to_categorical(label_indices, num_classes=len(label_to_id))
    label_inputs = np.concatenate([label_inputs, label_inputs])
    return train_inputs, label_inputs


def generates_batch(x, y, batch_size=64):
    '''
    :param x: train or valid matrix
    :param y: label matrix
    :param batch_size:
    :return: batch data after shuffle
    '''

    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for index in range(num_batch):
        start = index * batch_size
        end = min((index + 1) * batch_size, data_len)
        yield x_shuffle[start:end], y_shuffle[start:end]

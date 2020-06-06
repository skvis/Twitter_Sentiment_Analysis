import config
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib


def read_csv():
    df = pd.read_csv(config.TRAIN_FILE, names=config.COLS, encoding=config.ENCODE)
    df = df.sample(frac=1).reset_index(drop=True)
    labels = list(df['target'])
    sentences = list(df['text'])
    return sentences, labels


def tokenizer_sequences():
    sentences, labels = read_csv()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences,
                           maxlen=config.MAX_LENGTH,
                           padding=config.PAD_TYPE,
                           truncating=config.TRUNC_TYPE)

    split = int(len(sentences) * config.VALID_PORTION)

    joblib.dump(tokenizer, f'{config.MODEL_PATH}tokenizer.pkl')
    joblib.dump(word_index, f"{config.MODEL_PATH}word_ind.pkl")
    joblib.dump(vocab_size, f"{config.MODEL_PATH}vocab_size.pkl")

    return padded, sentences, labels, split


def split_dataset():
    padded, sentences, labels, split = tokenizer_sequences()

    train_padded = padded[split:len(sentences)]
    train_labels = labels[split:len(sentences)]
    valid_padded = padded[0:split]
    valid_labels = labels[0:split]

    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)

    return train_padded, train_labels, valid_padded, valid_labels


def read_test_csv():
    df = pd.read_csv(config.TEST_FILE, names=config.COLS, encoding=config.ENCODE)
    labels = list(df['target'])
    sentences = list(df['text'])
    return sentences, labels


def padded_test():
    tokenizer = joblib.load(f'{config.MODEL_PATH}tokenizer.pkl')
    sentences, labels = read_test_csv()
    test_sequences = tokenizer.texts_to_sequences(sentences)
    test_padded = pad_sequences(test_sequences,
                                maxlen=config.MAX_LENGTH,
                                padding=config.PAD_TYPE,
                                truncating=config.TRUNC_TYPE)
    test_labels = np.array(labels)

    return test_padded, test_labels


if __name__ == '__main__':
    split_dataset()

# Download the dataset
# https://docs.google.com/file/d/0B04GJPshIjmPRnZManQwWEdTZjg/edit
# Unzip the dataset

# import library

# Dataset
import pandas as pd

# utility
import os
import random
import numpy as np

# Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


# Dataset
cols = ['target', 'ids', 'data', 'flag', 'user', 'text']
encode = 'ISO-8859-1'

# tensorflow
oov_tok = '<00V>'
max_length = 16
trunc_type = 'post'
pad_type = 'post'
embedding_dim = 100
valid_portion = 0.1

data_path = 'input/trainingandtestdata/'
train_file = os.path.join(data_path, 'training.1600000.processed.noemoticon.csv')
test_file = os.path.join(data_path, 'testdata.manual.2009.06.14.csv')

df = pd.read_csv(train_file, encoding=encode, names=cols)
# print(df.head())
# print(df['target'].nunique())
df = df.sample(frac=1).reset_index(drop=True)
labels = list(df['target'])
sentences = list(df['text'])
# print(len(sentences))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
vocab_size = len(word_index)
# print(vocab_size)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences,
                       maxlen=max_length,
                       padding=pad_type,
                       truncating=trunc_type)

split = int(len(sentences) * valid_portion)
# print(split)

train_padded = padded[split:len(sentences)]
train_labels = labels[split:len(sentences)]
valid_padded = padded[0:split]
valid_labels = labels[0:split]

# print(type(train_padded), type(train_labels))
# print(len(train_padded), len(train_labels))
# print(len(valid_padded), len(valid_labels))

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

# Download the Glove Vector
# unzip it
# load the 100d

embeddings_index = {}
with open('input/glove.6B/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim))
for word, ind in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[ind] = embedding_vector

model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size+1,
                                      embedding_dim,
                                      input_length=max_length,
                                      weights=[embeddings_matrix],
                                      trainable=False),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv1D(64, 5, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

num_epochs = 2
BATCH_SIZE = 64
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

history = model.fit(train_padded, train_labels,
                    validation_data=(valid_padded, valid_labels),
                    epochs=num_epochs,
                    batch_size=BATCH_SIZE
                    verbose=2)  

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training Vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

plt.figure()
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training Vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss', 'Validation Loss'])
plt.show()    

# Evaluate

df_test = pd.read_csv(test_file, encoding=encode, names=cols)
test_labels = list(df_test['target'])
test_sentences = list(df_test['text'])

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences,
                       maxlen=max_length,
                       padding=pad_type,
                       truncating=trunc_type)

test_labels = np.array(test_labels)
score = model.evaluate(test_padded, test_labels, batch_size=16)

print('Accuracy:', score[1])
print('Loss:', score[0])


def predict(text):
    start_at = time.time()
    predict_text = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_length)
    score = model.predict([predict_text])[0]
    label = 'positive' if score >= 0.5 else 'negative'
    return {'label': label, 'score':float(score),
            'elapsed_time:':time.time()-start_at}

predict('I love the music')
            
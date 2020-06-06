import config
import utils
import tensorflow as tf
import joblib


def rnn():
    _ = utils.create_embed()
    vocab_size = joblib.load(f'{config.MODEL_PATH}vocab_size.pkl')
    embeddings_matrix = joblib.load(f'{config.MODEL_PATH}embed_matrix.pkl')

    model = tf.keras.Sequential([
                tf.keras.layers.Embedding(vocab_size+1,
                                          config.EMBEDDING_DIM,
                                          input_length=config.MAX_LENGTH,
                                          weights=[embeddings_matrix],
                                          trainable=False),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Conv1D(64, 5, activation='relu'),
                tf.keras.layers.MaxPooling1D(pool_size=4),
                tf.keras.layers.LSTM(64),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(1, activation='sigmoid')])

    return model

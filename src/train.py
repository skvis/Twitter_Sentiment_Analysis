from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import model
import config
import data_preprocess
import joblib
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def train_fn():

    train_padded, train_labels, valid_padded, valid_labels = data_preprocess.split_dataset()

    rnn_model = model.rnn()
    # print(model.summary())

    rnn_model.compile(optimizer=Adam(lr=config.LEARNING_RATE),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
                 EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

    history = rnn_model.fit(train_padded, train_labels,
                            validation_data=(valid_padded, valid_labels),
                            epochs=config.NUM_EPOCHS,
                            batch_size=config.BATCH_SIZE,
                            verbose=2,
                            callbacks=callbacks)

    rnn_model.save(f"{config.MODEL_PATH}my_model.h5")
    np.save(f'{config.MODEL_PATH}my_history.npy', history.history)


if __name__ == '__main__':
    train_fn()

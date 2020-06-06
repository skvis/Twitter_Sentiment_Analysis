import config
import data_preprocess
import joblib
import tensorflow as tf
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences


def evaluate(load_model):
    test_padded, test_labels = data_preprocess.padded_test()
    score = load_model.evaluate(test_padded, test_labels)

    print('Accuracy:', score[1])
    print('Loss:', score[0])


def predict(load_model, tokenizer, text):
    start_at = time.time()
    predict_text = pad_sequences(tokenizer.texts_to_sequences([text]),
                                 maxlen=config.MAX_LENGTH)
    score = load_model.predict([predict_text])[0]
    label = 'positive' if score >= 0.5 else 'negative'
    return {'label': label, 'score': float(score),
            'elapsed_time:': time.time()-start_at}


if __name__ == '__main__':
    tokenizer = joblib.load(f'{config.MODEL_PATH}tokenizer.pkl')
    load_model = tf.keras.models.load_model(f"{config.MODEL_PATH}my_model.h5")
    evaluate(load_model)

    prediction = predict(load_model, tokenizer, 'I love listening songs')
    print(prediction)

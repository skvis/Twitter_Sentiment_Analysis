import config
import numpy as np
import joblib


def create_embed():
    word_index = joblib.load(f'{config.MODEL_PATH}word_ind.pkl')
    vocab_size = joblib.load(f'{config.MODEL_PATH}vocab_size.pkl')

    embeddings_index = {}
    with open(config.GLOVE_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((vocab_size+1, config.EMBEDDING_DIM))
    for word, ind in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[ind] = embedding_vector

    joblib.dump(embeddings_matrix, f"{config.MODEL_PATH}embed_matrix.pkl")
    return embeddings_matrix


if __name__ == '__main__':
    create_embed()

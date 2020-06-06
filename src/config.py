import os

DATA_PATH = '../input/trainingandtestdata'
MODEL_PATH = '../models/'
TRAIN_FILE = os.path.join(DATA_PATH, 'training.1600000.processed.noemoticon.csv')
TEST_FILE = os.path.join(DATA_PATH, 'testdata.manual.2009.06.14.csv')
GLOVE_PATH = '../input/glove/glove.6B.100d.txt'

COLS = ['target', 'ids', 'data', 'flag', 'user', 'text']
ENCODE = 'ISO-8859-1'
VALID_PORTION = 0.1

# VOCAB_SIZE = 10000
# OOV_TOKEN = "<00V>"
EMBEDDING_DIM = 100
MAX_LENGTH = 16
TRUNC_TYPE = 'post'
PAD_TYPE = 'post'

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2

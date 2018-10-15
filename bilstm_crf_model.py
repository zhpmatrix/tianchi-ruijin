from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
import process_data
import pickle



def create_model(embed_dim, birnn_units,train=True):
    
    if train:
        (train_x, train_y), (test_x, test_y), (word2idx, chunk_tags) = process_data.load_data()
    else:
        with open('data/dict.pkl', 'rb') as inp:
            (word2idx, chunk_tags) = pickle.load(inp)
    
    vocab = [word for word, idx in word2idx.items()]

    model = Sequential()
    model.add(Embedding(len(vocab), embed_dim, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(birnn_units // 2, return_sequences=True)))
    
    # default learn mode is 'join'
    crf = CRF(len(chunk_tags), sparse_target=True)
    #crf = CRF(len(chunk_tags), sparse_target=True, learn_mode='marginal', test_mode='marginal')
    
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    
    if train:
        return model, (train_x, train_y), (test_x, test_y)
    else:
        return model, (word2idx, chunk_tags)

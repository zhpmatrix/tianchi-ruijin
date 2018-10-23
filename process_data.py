import numpy
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
import pickle
import platform


def load_data():
    train = _parse_data(open('data/ruijin_train.data', 'rb'))
    test = _parse_data(open('data/ruijin_dev.data', 'rb'))
    
    word_counts = Counter(row[0] for sample in train+test for row in sample)
    
    vocab = [w for w, f in iter(word_counts.items()) if f >= 2]
    
    word2idx = dict((w, i+2) for i, w in enumerate(vocab))
    
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    
    chunk_tags = ['O', 'B-Disease', 'I-Disease', 'B-Reason', 'I-Reason', "B-Symptom", "I-Symptom", "B-Test", "I-Test", "B-Test_Value", "I-Test_Value", "B-Drug", "I-Drug", "B-Frequency", "I-Frequency", "B-Amount", "I-Amount", "B-Treatment", "I-Treatment", "B-Operation", "I-Operation", "B-Method", "I-Method", "B-SideEff","I-SideEff","B-Anatomy", "I-Anatomy", "B-Level", "I-Level", "B-Duration", "I-Duration"]

    with open('data/dict.pkl', 'wb') as outp:
        pickle.dump((word2idx,chunk_tags), outp)

    train = _process_data(train, word2idx, chunk_tags)
    test = _process_data(test, word2idx, chunk_tags)
    return train, test, (word2idx, chunk_tags)


def _parse_data(fh):
    #  in windows the new line is '\r\n\r\n' the space is '\r\n' . so if you use windows system,
    #  you have to use recorsponding instructions

    if platform.system() == 'Windows':
        split_text = '\r\n'
    else:
        split_text = '\n'

    string = fh.read().decode('utf-8')
    data = [[row.split() for row in sample.split(split_text)] for
            sample in
            string.strip().split(split_text + split_text)]
    fh.close()
    return data


def _process_data(data, word2idx, chunk_tags, maxlen=None, onehot=False):
    if maxlen is None:
        maxlen = max(len(s) for s in data)
    
    
    x = [[word2idx.get(w[0], 1) for w in s] for s in data]

    y_chunk = [[chunk_tags.index(w[1]) for w in s] for s in data]
    
    # debug 
    #y_chunk = []
    #for i,s in enumerate(data):
    #    for j,w in enumerate(s):
    #        print(i,j)
    #        y_chunk.append(chunk_tags.index(w[1]))

    x = pad_sequences(x, maxlen)  # left padding

    y_chunk = pad_sequences(y_chunk, maxlen, value=-1)

    if onehot:
        y_chunk = numpy.eye(len(chunk_tags), dtype='float32')[y_chunk]
    else:
        y_chunk = numpy.expand_dims(y_chunk, 2)
    return x, y_chunk


def process_data(data, word2idx, maxlen=2000):
    x = [word2idx.get(w[0], 1) for w in data]
    length = len(x)
    x = pad_sequences([x], maxlen)  # left padding
    return x, length

if __name__ == '__main__':
    load_data()

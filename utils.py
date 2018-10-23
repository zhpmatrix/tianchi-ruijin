#coding=utf-8

from __future__ import print_function
import os
import pandas as pd

def gpu_config(gpu_num):
    
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    KTF.set_session(sess)
    print('GPU config done!')

def get_train_data(data_dir,cv_ratio=0.1):
    texts    = []
    tags     = []
    
    fileidxs = set()
    for filename in os.listdir(data_dir):
        fileidxs.add( filename.split('.')[0] )
    
    for fileidx in fileidxs:
        
        with open(data_dir+fileidx+'.txt', 'rb') as f:
            text = f.read().decode('utf-8')
        text_list = [char for char in text]
        
        tag = pd.read_csv(data_dir+fileidx+'.ann', header=None, sep='\t')
        tag_list = ['O' for _ in range( len(text_list) )]
        
        for i in range(tag.shape[0]):
            tag_item = tag.iloc[i][1].split(' ')
            cls, start, end = tag_item[0], int( tag_item[1] ), int( tag_item[-1] )
            
            tag_list[start] = 'B-'+cls
            for j in range(start+1, end):
                tag_list[j] = 'I-'+cls
        assert(len(text_list) == len(tag_list))
        texts.append(text_list)
        tags.append(tag_list)
    
    # write data into file
    split_chars = ['。', '！', '？', '，']
    train_num   = 0
    dev_num     = 0
    doc_dev_num = int(len(texts) * cv_ratio)

    train_file = 'data/ruijin_train.data'
    if os.path.exists(train_file):
        os.remove(train_file)

    with open(train_file, 'a') as f:
        for k in range(len(texts) - doc_dev_num):
            text_ = texts[k]
            tag_  = tags[k]
            for p in range(len(text_)):
                if text_[p] == '\n':
                    f.write('LB'+'\t'+tag_[p]+'\n')
                elif text_[p] == ' ':
                    f.write('SPACE'+'\t'+tag_[p]+'\n')
                elif text_[p] in split_chars:
                    train_num += 1
                    f.write(text_[p]+'\t'+tag_[p]+'\n\n')
                else:
                    f.write(text_[p]+'\t'+tag_[p]+'\n')
    
    dev_file = 'data/ruijin_dev.data'
    if os.path.exists(dev_file):
        os.remove(dev_file)
    with open(dev_file, 'a') as f:
        for k in range(len(texts) - doc_dev_num, len(texts)):
            text_ = texts[k]
            tag_  = tags[k]
            for p in range(len(text_)):
                if text_[p] == '\n':
                    f.write('LB'+'\t'+tag_[p]+'\n')
                elif text_[p] == ' ':
                    f.write('SPACE'+'\t'+tag_[p]+'\n')
                elif text_[p] in split_chars:
                    dev_num += 1
                    f.write(text_[p]+'\t'+tag_[p]+'\n\n')
                else:
                    f.write(text_[p]+'\t'+tag_[p]+'\n')
    print('train_num:{}, dev_num:{}'.format(train_num, dev_num))
    return train_num, dev_num

if __name__ == '__main__':
    train_num, dev_num = get_train_data(data_dir='data/raw/train/')

import bilstm_crf_model
import process_data
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import classification_report
from utils import *
from tqdm import tqdm

def get_test_data(txt_data, tag_data):
    with open(txt_data,'rb') as f:
        predict_text = f.read().decode('utf-8')

    predict_list = [char for char in predict_text]
    char_num     = len(predict_list)

    tag = pd.read_csv(tag_data, header=None, sep='\t')
    tag_list = ['O' for _ in range( char_num )]
    for i in range(tag.shape[0]):
        tag_item = tag.iloc[i][1].split(' ')
        cls, start, end = tag_item[0], int( tag_item[1] ), int( tag_item[-1] )
        tag_list[start] = 'B-'+cls
        for j in range(start+1, end):
            tag_list[j] = 'I-'+cls
    return predict_text, tag_list


def show(predict_text, result_tags, cls_name='Disease'):
    result = ''
    for s, t in zip(predict_text, result_tags):
        if t in ('B-'+cls_name, 'I-'+cls_name):
            result += ' ' + s if (t == 'B-'+cls_name) else s
    print([cls_name+':' + result])

def local_test(txt_data, tag_data, model, word2idx, chunk_tags):
    predict_text, tag_list  = get_test_data(txt_data, tag_data)
    str_, length = process_data.process_data(predict_text, word2idx, len(tag_list))
    raw = model.predict(str_)[0][-length:]
    result = [np.argmax(row) for row in raw]
    result_tags = [chunk_tags[i] for i in result]
    print(classification_report(tag_list, result_tags))
    #show(predict_text, result_tags, "Drug")

def savefile(tag_data,result_tags, predict_text):
    
    tags = []
    for tag in result_tags:
        if tag != 'O':
            tag_ = tag.split('-')[1]
        else:
            tag_ = tag
        tags.append(tag_)
    
    # write here
    prev = tags[0]
    start = 0
    num = 0
    for i in range(1, len(tags)):
        cur = tags[i]
        if cur != prev:
            end = i-1
            if prev != 'O':
                num += 1
                content = predict_text[start:end]
                content = content.replace('\n',' ')
                tag_data.write('T'+str(num)+'\t'+prev+' '+str(start)+' '+str(end)+'\t'+content+'\n') 
            start = i
            prev = cur
    tag_data.close()

def test(test_dir, submit_dir, model, word2idx, chunk_tags):
    for filename in tqdm( os.listdir(test_dir) ):
        fileidx = filename.split('.')[0]
        txt_data = test_dir + filename
        tag_data = open(submit_dir+fileidx+'.ann','w')
        with open(txt_data,'rb') as f:
            predict_text = f.read().decode('utf-8')
        str_, length = process_data.process_data(predict_text, word2idx, len(predict_text))
        raw = model.predict(str_)[0][-length:]
        result = [np.argmax(row) for row in raw]
        result_tags = [chunk_tags[i] for i in result]
        savefile(tag_data, result_tags, predict_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAL')
    parser.add_argument('--num', type=int)
    parser.add_argument('--embed', type=int)
    parser.add_argument('--units', type=int)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    gpu_config(args.gpu)
    
    # load model
    model_dir = 'expr/'+str(args.num)+'/model.h5'
    model, (word2idx, chunk_tags) = bilstm_crf_model.create_model(args.embed, args.units, train=False)
    model.load_weights(model_dir)
    
    test_dir = 'data/raw/test/'
    submit_dir = 'data/raw/submit/'
    #test(test_dir, submit_dir, model,word2idx, chunk_tags)

    
    txt_data =  'data/raw/local_test/152_6.txt'
    tag_data =  'data/raw/local_test/152_6.ann'
    local_test(txt_data, tag_data,model, word2idx, chunk_tags)

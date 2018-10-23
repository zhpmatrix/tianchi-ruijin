#coding=utf-8

from __future__ import print_function
import os
import pandas as pd
from tqdm import tqdm

def data_checker(data_dir):
    tags = ['B-Disease', 'I-Disease', 'B-Reason', 'I-Reason', "B-Symptom", "I-Symptom", "B-Test", "I-Test", "B-Test_Value", "I-Test_Value", "B-Drug", "I-Drug", "B-Frequency", "I-Frequency", "B-Amount", "I-Amount", "B-Treatment", "I-Treatment", "B-Operation", "I-Operation", "B-Method", "I-Method", "B-SideEff","I-SideEff","B-Anatomy", "I-Anatomy", "B-Level", "I-Level", "B-Duration", "I-Duration"]
    tag_set = {k.split('-')[1]:set() for k in tags}
    tag_num = {k.split('-')[1]:0 for k in tags}

    fileidxs = set()
    for filename in os.listdir(data_dir):
        fileidxs.add( filename.split('.')[0] )
    
    for fileidx in tqdm(fileidxs):
        tag = pd.read_csv(data_dir+fileidx+'.ann', header=None, sep='\t')
        for i in range(tag.shape[0]):
            cls, content = tag.iloc[i][1].split(' ')[0], tag.iloc[i][2]
            tag_num[cls] += 1
            tag_set[cls].add(content)  
    get_dict('./dict/',tag_set)
    return tag_set

def get_dict(word_dir,tag_dict):
    for k, v in tag_dict.items():
        with open(word_dir+k+'.txt', 'w') as f:
            for word in v:
                f.write(word+'\n')

def seg_test(filepath,cwspath,dictpath):
    from pyltp import Segmentor
    segmentor = Segmentor()
    segmentor.load_with_lexicon(cwspath, dictpath)
    
    text = open(filepath).read()
    words = segmentor.segment(text)
    print('\t'.join(words))
    segmentor.release()

if __name__ == '__main__':
    seg_test('../data/raw/train/9.txt', './model/cws.model','./dict/Disease.txt')
    #data_checker(data_dir='../data/raw/train/')

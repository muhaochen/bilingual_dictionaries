from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import scipy
import numpy as np
import tqdm
import tensorflow as tf

import multiprocessing
from multiprocessing import Process, Value, Lock, Manager

# tester for any model

if '../src' not in sys.path:
    sys.path.append('../src')

from dict2l import dict2l
from dict2l import safe_lower

data = dict2l()

en_stop = data.load_stopwords('../../../data/stop_words_en.txt')
fr_stop = data.load_stopwords('../../../data/stop_words_fr.txt')
es_stop = data.load_stopwords('../../../data/stop_words_es.txt')


TopK = 100

lan1 = 'en'
lan2 = 'fr'

filepath = '../../../data/final/4/fr_en_test500.csv'

model_name = './en_fr_gru.h5'

ofile = 'en_fr_gru.txt'

ds_files12 = []
ds_files21 = []

d_len = 18

if len(sys.argv) > 1:
    assert(len(sys.argv) >= 7 and len(sys.argv) % 2 == 1)
    if len(sys.argv) == 7:
        d_len, lan1, lan2, filepath, model_name, ofile = sys.argv[1:]
    else:
        d_len,lan1, lan2, filepath, model_name, ofile = sys.argv[1:7]
        gap = int((len(sys.argv) - 6) / 2)
        ds_files12 = sys.argv[7:7+gap]
        ds_files21 = sys.argv[7+gap:]

d_len = int(d_len)
data.set_length(d_len)

if lan1 != lan2:
    if 'fr' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-fr.en.50.txt', en_stop, 'en', ' ')
        data.load_word2vec('../../bilbowa/en-fr.fr.50.txt', fr_stop, 'fr', ' ')
    elif 'es' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-es.en.50.txt', en_stop, 'en', ' ')
        data.load_word2vec('../../bilbowa/en-es.es.50.txt', es_stop, 'es', ' ')
else:
    if 'fr' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-fr.fr.50.txt', fr_stop, 'fr', ' ')
    elif 'es' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-es.es.50.txt', es_stop, 'es', ' ')
    else:
        data.load_word2vec('../../bilbowa/en-fr.en.50.txt', en_stop, 'en', ' ')

f_dedup = set([])
data.word_desc_pool(filepath, lan1+lan2+'test', lower=True, src_lan = lan1, tgt_lan = lan2, splitter=',', save_desc=True)
if len(ds_files12) > 0:
    for f in ds_files12:
        if f not in f_dedup:
            data.load_vocab_set(f, lan1, lan2, lan2_only=True)
            f_dedup.add(f)
    for f in ds_files21:
        if f not in f_dedup:
            data.load_vocab_set(f, lan2, lan1, lan2_only=True)
            f_dedup.add(f)

test_d, test_w = [], []
for d, w in data.dw_pair[lan1+lan2+'test']:
    test_d.append(data.desc_embed[lan1+lan2+'test'][d])
    test_w.append(w)

test_d = np.array(test_d)
test_w = np.array(test_w)

## --- keras

def match(tk1, tk2):
    return tk1 == tk2
    #return tk1.find(tk2) > -1 or tk2.find(tk1) > -1

def hits(plist, t):
    hit = 0.
    token_t = data.tokens[lan2][t]
    rst = []
    for l, d in plist:
        if hit > 0.:
            rst.append(hit)
            continue
        token_l = safe_lower(data.tokens[lan2][l])
        #if token_l.find(token_t) > -1 or token_t.find(token_l) > -1:
        if match(token_l, token_t):
            hit = 1.
        rst.append(hit)
    return np.array(rst)

def rank(vec, t, vocab_set = True):
    token_t = data.tokens[lan2][t]
    rank = 1
    dist_l = data.distance(data.w2v(token_t, lan2), vec)
    if vocab_set:
        cand_space = data.cand_vocab[lan2]
    else:
        cand_space = range(1, len(data.tokens[lan2]))
    for i in cand_space:
        if match(data.tokens[lan2][i], token_t):
            dist_c = data.distance(data.i2v(i, lan2), vec)
            if dist_c < dist_l:
                dist_l = dist_c
    for i in range(1, len(data.tokens[lan2])):
        if data.distance(data.i2v(i, lan2), vec) < dist_l:
            rank += 1
    return rank

def aggr(preds, tgts, topk = 100):
    assert (len(preds) == len(tgts))
    ht = np.zeros(topk)
    mrr = []
    for i in tqdm.tqdm(range(len(tgts))):
        plist = data.kNN(preds[i], lan2, topk, limit_ids=data.cand_vocab[lan2])
        ht += hits(plist, tgts[i])
        mrr.append(1. / rank(preds[i], tgts[i]))
    return ht / len(tgts), np.mean(mrr)

def aggr_mp(preds, tgts, index, ht_list, mr_list, topk = 100):
    while index.value < len(preds):
        i = index.value
        index.value += 1
        plist = data.kNN(preds[i], lan2, topk, limit_ids=data.cand_vocab[lan2])
        if i < 10:
            print(plist[:10])
        ht_list.append(hits(plist, tgts[i]))
        mr_list.append(1. / rank(preds[i], tgts[i], True))
        if (i % 25 == 1):
            print(i)

cpu_count = multiprocessing.cpu_count()

manager = Manager()

index = Value('i', 0, lock=True) #index    

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, GRU, Bidirectional, Merge, BatchNormalization, merge
from keras.layers.core import Flatten, Reshape
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.layers import Input
from utils_loss import cosine_loss, l2_hinge, cosine_hinge, l2_dist

fp = open(ofile, 'w')

batchsize1 = 50

model = keras.models.load_model(model_name, custom_objects={'cosine_loss': cosine_loss, 'l2_hinge': l2_hinge, 'l2_dist': l2_dist, 'cosine_hinge': cosine_hinge, 'length': d_len})
pred = model.predict(test_d)
assert (len(pred) == len(test_w))
ht_list = manager.list()
mrr = manager.list()

processes = [Process(target=aggr_mp, args=(pred, test_w, index, ht_list, mrr, TopK)) for x in range(cpu_count - 1)]
for p in processes:
    p.start()
for p in processes:
    p.join()

ht = np.zeros(TopK)

for x in ht_list:
    ht += x
ht /= len(pred)
mrr = np.mean(mrr)
    
print(ht[10], ht[-1])
print(mrr)
fp.write('hits\n' + ' '.join([str(x) for x in ht]) + '\nmrr=' + str(mrr))

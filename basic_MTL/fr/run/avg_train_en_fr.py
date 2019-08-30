from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import numpy as np

if '../src' not in sys.path:
    sys.path.append('../src')

from dict2l import dict2l

data = dict2l()

en_stop = data.load_stopwords('../../../data/stop_words_en.txt')
fr_stop = data.load_stopwords('../../../data/stop_words_fr.txt')
es_stop = data.load_stopwords('../../../data/stop_words_es.txt')

data.set_length(15)

filepath = '../../../data/final/4/fr_en_train500_4.csv'

model_name = './en_fr_gru.h5'

lan1 = 'en'
lan2 = 'fr'

if len(sys.argv) > 1:
    assert(len(sys.argv) == 5)
    lan1, lan2, filepath, model_name = sys.argv[1:]

if lan1 != lan2:
    if 'fr' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-fr.en.50.txt', en_stop, 'en', ' ')
        data.load_word2vec('../../bilbowa/en-fr.fr.50.txt', fr_stop, 'fr', ' ')
    if 'es' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-es.en.50.txt', en_stop, 'en', ' ')
        data.load_word2vec('../../bilbowa/en-es.es.50.txt', es_stop, 'es', ' ')
else:
    if 'en' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-fr.en.50.txt', en_stop, 'en', ' ')
    elif 'fr' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-fr.fr.50.txt', fr_stop, 'fr', ' ')
    elif 'es' in [lan1, lan2]:
        data.load_word2vec('../../bilbowa/en-fr.en.50.txt', en_stop, 'en', ' ')
        data.load_word2vec('../../bilbowa/en-es.es.50.txt', es_stop, 'es', ' ')

data.word_desc_pool(filepath, lan1+lan2+'train', lower=True, src_lan = lan1, tgt_lan = lan2, splitter=',', save_desc=True)

train_d, train_w = [], []
for d, w in data.dw_pair[lan1+lan2+'train']:
    train_d.append(data.desc_embed[lan1+lan2+'train'][d])
    train_w.append(data.wv[lan2][w])

train_d = np.array(train_d)
train_w = np.array(train_w)

length = data.length
dim = data.dim
batchsize1 = 50


## --- keras

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, GRU, Bidirectional, Merge, BatchNormalization, merge, Average
from keras.layers.core import Flatten, Reshape
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.layers import Input, GlobalAveragePooling1D
from keras import backend as K

def build_model():
    desc_input = Input(shape=(length, dim), name='title1')
    x=GlobalAveragePooling1D()(desc_input)
    output=Dense(dim, activation=None, use_bias=False)(Dense(dim, activation=None, use_bias=False)(Dense(dim, activation=None, use_bias=False)(x)))
    model = Model(inputs=[desc_input], outputs=[output])
    return model

adam = Adam(lr=0.00025, beta_1=0.9, amsgrad=True)
model = build_model()
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
model.fit(train_d, train_w, batch_size=batchsize1, epochs=100)
model.save(model_name)
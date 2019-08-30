from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import csv
import numpy as np
import scipy
import tensorflow as tf
from keras import backend as K

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
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, CuDNNGRU, Bidirectional, Merge, BatchNormalization, merge
from keras.layers.core import Flatten, Reshape
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import Adam
from keras.layers import Input

def build_model():
    desc_input = Input(shape=(length, dim), name='title1')
    x=CuDNNGRU(3*dim, return_sequences=True)(desc_input)
    #x=MaxPooling1D(2)(x)
    x=CuDNNGRU(3*dim, return_sequences=True)(x)
    #x=MaxPooling1D(2)(x)
    x=CuDNNGRU(3*dim, return_sequences=True)(x)
    x=CuDNNGRU(3*dim, return_sequences=True)(x)
    #x=MaxPooling1D(2)(x)
    x=CuDNNGRU(3*dim, return_sequences=True)(x)
    x=CuDNNGRU(2*dim)(x)
    output=Dense(dim, activation=None)(Dense(dim, activation=None)(x))
    model = Model(inputs=[desc_input], outputs=[output])
    return model

def cosine_loss(y_true, y_pred):
    #return tf.losses.cosine_distance(y_true, y_pred, axis=-1)
    a = tf.divide(y_true, tf.norm(y_true, axis=-1, keepdims=True))
    b = tf.divide(y_pred, tf.norm(y_pred, axis=-1, keepdims=True))
    #loss = 1. - tf.reduce_sum(tf.multiply(a, b), axis=-1)
    #return tf.reduce_mean(loss, axis=-1)
    return tf.losses.cosine_distance(a, b, axis=-1)

def l2_dist(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=-1))

adam = Adam(lr=0.00025, beta_1=0.9, amsgrad=True)
model = build_model()
#model.compile(optimizer='adam', loss=cosine_loss, metrics=[cosine_loss])
model.compile(optimizer=adam, loss='mse', metrics=['mse'])
model.fit(train_d, train_w, batch_size=batchsize1, epochs=400)
model.save(model_name)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import csv
import scipy
import heapq as HP
import tqdm
import scipy
import tensorflow as tf
from keras import backend as K
from random import randint

batchsize1 = 50

def cosine_loss(y_true, y_pred):
    a = tf.divide(y_true, tf.norm(y_true, axis=-1, keepdims=True))
    b = tf.divide(y_pred, tf.norm(y_pred, axis=-1, keepdims=True))
    return tf.losses.cosine_distance(a, b, axis=-1)

def l2_hinge(y_true, y_pred):
    margin = 0.5
    ind = []
    bound = batchsize1
    for i in range(bound):
        x = randint(0, bound - 1)
        while x == i:
            x = randint(0, bound - 1)
        ind.append(x)
    ind = tf.constant(ind, dtype = tf.int32)
    y_neg = tf.nn.embedding_lookup(y_true, ind)
    #y_neg = tf.random_shuffle(y_true)
    l_pos = tf.norm(y_true - y_pred, axis=-1)
    l_neg = tf.norm(y_neg - y_pred, axis=-1)
    return tf.reduce_mean(tf.maximum(l_pos - l_neg + margin, 0.))

def l2_dist(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=-1))

def cosine_hinge(y_true, y_pred):
    margin = 0.2
    a = tf.divide(y_true, tf.norm(y_true, axis=-1, keepdims=True))
    b = tf.divide(y_pred, tf.norm(y_pred, axis=-1, keepdims=True))
    ind = []
    #c = y_neg = tf.random_shuffle(a)
    bound = batchsize1
    for i in range(bound):
        x = randint(0, bound - 1)
        while x == i:
            x = randint(0, bound - 1)
        ind.append(x)
    ind = tf.constant(ind, dtype = tf.int32)
    c = tf.nn.embedding_lookup(a, ind)
    l_pos = tf.reduce_sum(tf.multiply(a, b), axis=-1)
    l_neg = tf.reduce_sum(tf.multiply(a, c), axis=-1)
    return tf.reduce_mean(tf.maximum(margin - l_pos + l_neg, 0.))

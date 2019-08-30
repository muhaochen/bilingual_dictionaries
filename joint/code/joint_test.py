#!/usr/bin/env python3

from collections import defaultdict
import math
import csv
from itertools import chain
import os
from os.path import join
import time

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import numpy as np
import heapq as HP

from sklearn.linear_model import LogisticRegression as LR
from sklearn.neural_network import MLPClassifier as MLP

import keras
from keras.optimizers import Adam

from data import Embedding, MultiLanguageEmbedding, \
    LazyIndexCorpus,  Word2vecIterator, BilbowaIterator, \
    DescCorpus, DescIterator
from model import get_joint_model, word2vec_loss, bilbowa_loss, \
encoder_loss

FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', '', 'root directory for data.')
flags.DEFINE_string('lang0_emb_file', '', '')
flags.DEFINE_string('lang1_emb_file', '', '')
flags.DEFINE_string('lang01_desc_file', '', '')
flags.DEFINE_string('lang10_desc_file', '', '')
flags.DEFINE_string('lang01_desc_test_file', '', '')
flags.DEFINE_string('lang10_desc_test_file', '', '')
flags.DEFINE_string('lang01_paraphrase_train_file', '', '')
flags.DEFINE_string('lang01_paraphrase_test_file', '', '')

flags.DEFINE_string('model_root', '', 'root directory for model')

flags.DEFINE_integer('word2vec_batch_size', 100000, '')
flags.DEFINE_integer('encoder_desc_length', 15, '')
flags.DEFINE_integer('encoder_batch_size', 50, '')


def main(argv):
    del argv  # Unused.

    # -- Load data

    # ---- Embedding
    logging.info('Loading embeddings')
    emb0 = Embedding(join(FLAGS.data_root, FLAGS.lang0_emb_file))
    emb1 = Embedding(join(FLAGS.data_root, FLAGS.lang1_emb_file))
    emb = MultiLanguageEmbedding(emb0, emb1)

    logging.info('Loadding desc - word pairs')
    # ---- desc, word pairs
    dw_pair_train_01 = DescCorpus.build_dw_pair_from_file(
        join(FLAGS.data_root, FLAGS.lang01_desc_file),
        emb,
        src_lan_id=0,
        tgt_lan_id=1)
    dw_pair_train_10 = DescCorpus.build_dw_pair_from_file(
        join(FLAGS.data_root, FLAGS.lang10_desc_file),
        emb,
        src_lan_id=1,
        tgt_lan_id=0)
    dw_pair_test_01 = DescCorpus.build_dw_pair_from_file(
        join(FLAGS.data_root, FLAGS.lang01_desc_test_file),
        emb,
        src_lan_id=0,
        tgt_lan_id=1)
    dw_pair_test_10 = DescCorpus.build_dw_pair_from_file(
        join(FLAGS.data_root, FLAGS.lang10_desc_test_file),
        emb,
        src_lan_id=1,
        tgt_lan_id=0)

    # ---- build candidate set
    lang0_candidate_set = []  # lang# is the lang_id of *target*
    for dw_pair in chain(dw_pair_train_10, dw_pair_test_10):
        d, w = dw_pair
        lang0_candidate_set.append(w)
    lang0_candidate_set = set(lang0_candidate_set)
    lang1_candidate_set = []  # lang# is the lang_id of *target*
    for dw_pair in chain(dw_pair_train_01, dw_pair_test_01):
        d, w = dw_pair
        lang1_candidate_set.append(w)
    lang1_candidate_set = set(lang1_candidate_set)

    # -- Load model(s)
    logging.info('Loading models.')
    tag = ''
    word2vec_model_infer = keras.models.load_model(
        join(FLAGS.model_root, tag + 'word2vec_model_infer'))
    encoder_model_infer = keras.models.load_model(
        join(FLAGS.model_root, tag + 'encoder_model_infer'))
    # encoder_model_infer.compile(optimizer=Adam(amsgrad=True), loss='mse')
    logging.info("Models are not compiled, but that's fine.")

    # -- Predicting
    logging.info("predicting...")
    # ---- Get embedding matrix
    emb_matrix = word2vec_model_infer.predict(
        np.arange(0, len(emb)), batch_size=FLAGS.word2vec_batch_size)
    logging.info('emb_matrix.shape = %s', emb_matrix.shape)

    # -- Test Task 1

    def run_test_task_1(dw_pair, lang_target_candidate_set):

        desc_iter = DescIterator(
            DescCorpus(dw_pair),
            desc_length=FLAGS.encoder_desc_length,
            batch_size=FLAGS.encoder_batch_size,
            shuffle=False,
            epochs=1,
        )
        desc_embedded = []
        for batch in desc_iter.iter(is_inference=True):
            r = encoder_model_infer.predict_on_batch(batch)
            desc_embedded.append(r)
        desc_embedded = np.concatenate(desc_embedded)
        desc_target = [_[1] for _ in dw_pair]

        def get_knn(embedded, emb_matrix, candidate_set):
            r = []
            for id_ in candidate_set:
                dist = np.linalg.norm(embedded - emb_matrix[id_])
                r.append((dist, id_))
            r.sort()
            return [_[1] for _ in r]

        def get_rank(target, candidates):
            rank = 0
            while candidates[rank] != target and rank + 1 < len(candidates):
                rank += 1
            assert candidates[rank] == target
            return rank + 1  # starting from 1

        hits_key = [1, 10, 100]
        hits_counter = defaultdict(int)
        total = 0
        mrr = []
        for embedeed, target in tqdm(
                zip(desc_embedded, desc_target), desc='test sents'):
            total += 1
            assert target in lang_target_candidate_set
            plist = get_knn(
                embedeed,
                emb_matrix,
                lang_target_candidate_set,
            )
            for hits in hits_key:
                hits_counter[hits] += int(target in plist[:hits])
            mrr.append(1.0 / get_rank(target, plist))

        hits_ratio = [1.0 * hits_counter[hits] / total for hits in hits_key]
        return hits_counter, hits_ratio, np.mean(mrr)

    lang_01_result = run_test_task_1(dw_pair_test_01, lang1_candidate_set)
    lang_10_result = run_test_task_1(dw_pair_test_10, lang0_candidate_set)

    print('TASK 1:')
    fout = open(join(FLAGS.model_root, 'test_task_1.txt'), 'w')

    def dual_print(*args):
        print(*args)
        print(*args, file=fout)

    dual_print('LANG 0 sent -> LANG 1 word')
    dual_print(lang_01_result[0])
    dual_print(lang_01_result[1])
    dual_print(lang_01_result[2])
    dual_print('LANG 1 sent -> LANG 0 word')
    dual_print(lang_10_result[0])
    dual_print(lang_10_result[1])
    dual_print(lang_10_result[2])

    fout.close()

    # -- Run test task 2 (paraphrasing)

    def desc_to_ids(desc, lang_id):
        r = emb.encode(
            [_.lower() for _ in desc.split()],
            lang_id=lang_id,
        )
        r = [_ for _ in r if _ != -1]
        return r

    def load_suite(filepath):
        lang0_d, lang1_d, target, n = [], [], [], 0
        for line in tqdm(
                csv.reader(
                    open(filepath, newline=''),
                    delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                ),
                desc=('Loading paraphrase from %s' % filepath),
        ):
            lang0_d.append(desc_to_ids(line[0], lang_id=0))
            lang1_d.append(desc_to_ids(line[1], lang_id=1))
            target.append(float(line[2]))
            n += 1

        desc_length = FLAGS.encoder_desc_length

        def get_embedded_sent(d):
            sent = np.zeros(shape=(n, desc_length), dtype=np.int32)
            mask = np.zeros(shape=(n, desc_length), dtype=np.float32)
            for i in range(n):
                this_d = d[i][:desc_length]
                lth = len(this_d)
                sent[i, :lth] = this_d
                mask[i, :lth] = 1.0
            embedded = encoder_model_infer.predict(
                [sent, mask], batch_size=FLAGS.encoder_batch_size)
            return embedded

        lang0_embedded = get_embedded_sent(lang0_d)
        lang1_embedded = get_embedded_sent(lang1_d)

        result_d, result_w, result_w1d = [], [], []
        for i in range(n):
            result_d.append([lang0_embedded[i], lang1_embedded[i]])
            result_w.append([target[i], 1 - target[i]])
            result_w1d.append(target[i])

        result_d = np.array(result_d)
        result_w = np.array(result_w)
        result_w1d = np.array(result_w1d, dtype='int32')

        # import pdb
        # pdb.set_trace()

        return result_d, result_w, result_w1d

    train_d, train_w, train_w1d = load_suite(
        join(FLAGS.data_root, FLAGS.lang01_paraphrase_train_file))

    test_d, _, test_w = load_suite(
        join(FLAGS.data_root, FLAGS.lang01_paraphrase_test_file))
    '''
    print('train_d.shape', train_d.shape)
    print('train_w.shape', train_w.shape)
    print('train_w1d.shape', train_w1d.shape)
    print('test_d.shape', test_d.shape)
    print('test_w.shape', test_w.shape)
    '''

    encode_train1 = train_d[:, 0]
    encode_train2 = train_d[:, 1]
    assert (len(encode_train1) == len(train_w))

    diff_train = np.array([
        encode_train1[i] - encode_train2[i] for i in range(len(encode_train1))
    ])
    dist_train = np.array([[np.linalg.norm(x)] for x in diff_train])

    encode_test1 = test_d[:, 0]
    encode_test2 = test_d[:, 1]

    diff_test = np.array(
        [encode_test1[i] - encode_test2[i] for i in range(len(encode_test1))])
    dist_test = np.array([[np.linalg.norm(x)] for x in diff_test])

    def match(pred, t):
        #if (len(pred.shape)) > 1:
        #pred = pred[0]
        if (pred > 0.5 and t < 0.5) or (pred < 0.5 and t > 0.5):
            return 0.
        return 1.

    logging.info('Fitting LR')
    logreg = LR()
    logreg.fit(dist_train, train_w1d)

    lr_rst = logreg.predict(dist_test)

    lr_accuracy = 0.
    for i in range(len(lr_rst)):
        lr_accuracy += match(lr_rst[i], test_w[i])
    lr_accuracy /= len(lr_rst)

    logging.info('fitting MLP')
    mlp = MLP(hidden_layer_sizes=[25, 13])

    mlp.fit(diff_train, train_w1d)

    mlp_rst = mlp.predict(diff_test)
    mlp_accuracy = 0.
    for i in range(len(mlp_rst)):
        mlp_accuracy += match(mlp_rst[i], test_w[i])
    mlp_accuracy /= len(mlp_rst)

    print('TASK 2:')
    fout = open(join(FLAGS.model_root, 'test_task_2.txt'), 'w')

    def dual_print(*args):
        print(*args)
        print(*args, file=fout)

    dual_print('LR accuracy', lr_accuracy)
    dual_print('MLP accuracy', mlp_accuracy)
    fout.close()


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

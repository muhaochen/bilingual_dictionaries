#!/usr/bin/env python3

import os
from os.path import join

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import numpy as np

from keras.optimizers import Adam

from data import Embedding, MultiLanguageEmbedding, DescCorpus
from model import encoder_ori_build_model

FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', '', 'root directory for data.')
flags.DEFINE_string('lang0_emb_file', '', '')
flags.DEFINE_string('lang1_emb_file', '', '')
flags.DEFINE_string('lang01_desc_file', '', '')
flags.DEFINE_string('lang10_desc_file', '', '')

flags.DEFINE_string('model_root', '', 'root directory for model')
flags.DEFINE_string('encoder_model_file', '', '')

flags.DEFINE_integer('encoder_desc_length', 15, '')
flags.DEFINE_integer('encoder_batch_size', 50, '')
flags.DEFINE_integer('encoder_epochs', 1000, '')
flags.DEFINE_float('encoder_lr', 0.0002, '')


def main(argv):
    del argv  # Unused.

    os.system('mkdir -p "%s"' % FLAGS.model_root)

    sw_emb0 = Embedding(join(FLAGS.data_root, FLAGS.lang0_emb_file))
    sw_emb1 = Embedding(join(FLAGS.data_root, FLAGS.lang1_emb_file))
    sw_emb = MultiLanguageEmbedding(sw_emb0, sw_emb1)

    desc_corpus = DescCorpus(
        DescCorpus.build_dw_pair_from_file(
            join(FLAGS.data_root, FLAGS.lang01_desc_file),
            sw_emb,
            src_lan_id=0,
            tgt_lan_id=1,
            debug_output_filepath=join(
                FLAGS.model_root,
                'build_dw_pair_from_file_lang01_desc_file.txt',
            ),
        ) + DescCorpus.build_dw_pair_from_file(
            join(FLAGS.data_root, FLAGS.lang10_desc_file),
            sw_emb,
            src_lan_id=1,
            tgt_lan_id=0,
            debug_output_filepath=join(
                FLAGS.model_root,
                'build_dw_pair_from_file_lang10_desc_file.txt',
            ),
        ), )

    length = FLAGS.encoder_desc_length
    dim = sw_emb.get_dim()

    train_d, train_w = desc_corpus.make_dw_embs(emb=sw_emb, length=length)
    logging.info('train_d.shape = %s', train_d.shape)
    logging.info('train_w.shape = %s', train_w.shape)

    model = encoder_ori_build_model(length=length, dim=dim)
    adam = Adam(lr=FLAGS.encoder_lr, beta_1=0.9, amsgrad=True)
    model.compile(optimizer=adam, loss='mse', metrics=['mse'])

    model.fit(
        train_d,
        train_w,
        batch_size=FLAGS.encoder_batch_size,
        epochs=FLAGS.encoder_epochs,
    )
    model.save(join(FLAGS.model_root, FLAGS.encoder_model_file))


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

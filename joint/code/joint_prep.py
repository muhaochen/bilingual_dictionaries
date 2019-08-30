#!/usr/bin/env python3

from collections import defaultdict
import os
from os.path import join

from absl import app
from absl import flags
from absl import logging
import numpy as np
from tqdm import tqdm
import yaml
from joblib import Parallel, delayed

from data import Embedding, MultiLanguageEmbedding, \
    line_to_id, lines_to_id

FLAGS = flags.FLAGS

flags.DEFINE_string('data_root', '', 'root directory for data.')
flags.DEFINE_string('lang0_emb_file', '', '')
flags.DEFINE_string('lang0_mono_string_corpus_file', '', '')
flags.DEFINE_string('lang0_multi_string_corpus_file', '', '')
flags.DEFINE_string('lang1_emb_file', '', '')
flags.DEFINE_string('lang1_mono_string_corpus_file', '', '')
flags.DEFINE_string('lang1_multi_string_corpus_file', '', '')

flags.DEFINE_integer('mono_max_lines', -1, '')
flags.DEFINE_integer('multi_max_lines', -1, '')

flags.DEFINE_string('model_root', '', 'root directory for model')
flags.DEFINE_string('lang0_mono_index_corpus_file', '', '')
flags.DEFINE_string('lang1_mono_index_corpus_file', '', '')
flags.DEFINE_string('lang0_multi_index_corpus_file', '', '')
flags.DEFINE_string('lang1_multi_index_corpus_file', '', '')


def work(string_corpus_file, index_corpus_file, emb, max_lines, lang_id):
    logging.info(
        'Converting %s -> %s',
        string_corpus_file,
        index_corpus_file,
    )
    fin = open(string_corpus_file)
    fout = open(index_corpus_file + '.ids.txt', 'w')
    fdebug = open(index_corpus_file + '.ids.debug.txt', 'w')
    dict_counts = defaultdict(int)
    vocab = emb.get_vocab()
    index = 0
    for raw_line in tqdm(fin, unit=' lines'):
        if max_lines >= 0 and index >= max_lines:
            break
        index += 1
        id_ = line_to_id(raw_line, emb, lang_id)
        for _ in id_:
            dict_counts[_] += 1
        print(' '.join(map(str, id_)), file=fout)
        if index <= 100:
            print(' '.join([vocab[_] for _ in id_]), file=fdebug)
            fdebug.flush()

    fin.close()
    fout.close()
    fdebug.close()

    logging.info('Preparing meta')
    max_id = max(dict_counts.keys())
    counts = [0] * (max_id + 1)
    for k, v in dict_counts.items():
        counts[k] = v
    eos_id = emb.encode('</s>', lang_id=lang_id)

    logging.info('Dumping meta')
    yaml.dump({'eos_id': eos_id}, open(index_corpus_file + '.meta.yaml', 'w'))

    logging.info('Dumping counts')
    np.savez(
        index_corpus_file + '.counts.npz', counts=np.array(counts, dtype='i'))

    logging.info('Finished')


def main(argv):
    del argv  # Unused.

    os.system('mkdir -p "%s"' % FLAGS.model_root)

    emb0 = Embedding(
        join(FLAGS.data_root, FLAGS.lang0_emb_file),
        keep_emb=False,
    )
    emb1 = Embedding(
        join(FLAGS.data_root, FLAGS.lang1_emb_file),
        keep_emb=False,
    )
    emb = MultiLanguageEmbedding(emb0, emb1)

    work(
        join(FLAGS.data_root, FLAGS.lang0_mono_string_corpus_file),
        join(FLAGS.model_root, FLAGS.lang0_mono_index_corpus_file),
        emb=emb,
        max_lines=FLAGS.mono_max_lines,
        lang_id=0,
    )
    work(
        join(FLAGS.data_root, FLAGS.lang1_mono_string_corpus_file),
        join(FLAGS.model_root, FLAGS.lang1_mono_index_corpus_file),
        emb=emb,
        max_lines=FLAGS.mono_max_lines,
        lang_id=1,
    )
    work(
        join(FLAGS.data_root, FLAGS.lang0_multi_string_corpus_file),
        join(FLAGS.model_root, FLAGS.lang0_multi_index_corpus_file),
        emb=emb,
        max_lines=FLAGS.multi_max_lines,
        lang_id=0,
    )
    work(
        join(FLAGS.data_root, FLAGS.lang1_multi_string_corpus_file),
        join(FLAGS.model_root, FLAGS.lang1_multi_index_corpus_file),
        emb=emb,
        max_lines=FLAGS.multi_max_lines,
        lang_id=1,
    )


import pdb, traceback, sys, code  # noqa

if __name__ == '__main__':
    try:
        app.run(main)
    except Exception:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

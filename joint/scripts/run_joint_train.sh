#!/usr/bin/env bash

#######################################
# Bash3 Boilerplate Start
# copied from https://kvz.io/blog/2013/11/21/bash-best-practices/

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

arg1="${1:-}"
# Bash3 Boilerplate End
#######################################

function en_fr (){
  MODEL_ROOT=../scratch/model/joint_en_fr/
  if [ ! -d "$MODEL_ROOT" ] ; then
    mkdir -p "$MODEL_ROOT"
    (cd "$MODEL_ROOT" && for f in `ls ../joint_en_fr/*` ; do ln -s $f . ; done )
  fi

  ./joint_train.py \
    --data_root ../scratch/data \
    --lang0_emb_file withctx.en-fr.en.50.1.txt \
    --lang1_emb_file withctx.en-fr.fr.50.1.txt \
    --lang0_ctxemb_file withctx.en-fr.en.50.1.txt.ctx \
    --lang1_ctxemb_file withctx.en-fr.fr.50.1.txt.ctx \
    --lang01_desc_file wiktionary_final/10/fr_en_train500_10.csv \
    --lang10_desc_file wiktionary_final/10/en_fr_train500_10.csv \
    --mono_max_lines -1 \
    --multi_max_lines -1 \
    --model_root "$MODEL_ROOT" \
    --lang0_mono_index_corpus_file en_mono \
    --lang1_mono_index_corpus_file fr_mono \
    --lang0_multi_index_corpus_file en_multi \
    --lang1_multi_index_corpus_file fr_multi \
    --emb_dim 50 \
    --word2vec_batch_size 100000 \
    --bilbowa_sent_length 50 \
    --bilbowa_batch_size 100 \
    --encoder_desc_length 15 \
    --encoder_batch_size 64 \
    --encoder_lr 0.0002 \
    --train_mono=true \
    --train_multi=true \
    --train_encoder=true \
    --max_encoder_epochs 1000 \
    --timing_scaling_mono 3.0 \
    --timing_scaling_multi 3.0 \
    --word_emb_trainable=true \
    --context_emb_trainable=true \
    --encoder_target_no_gradient=true \
    --encoder_arch_version=1 \
    ;
}

function en_es (){
  MODEL_ROOT=../scratch/model/joint_en_es/
  if [ ! -d "$MODEL_ROOT" ] ; then
    mkdir -p "$MODEL_ROOT"
    (cd "$MODEL_ROOT" && for f in `ls ../joint_en_es/*` ; do ln -s $f . ; done )
  fi

  ./joint_train.py \
    --data_root ../scratch/data \
    --lang0_emb_file withctx.en-es.en.50.1.txt \
    --lang1_emb_file withctx.en-es.es.50.1.txt \
    --lang0_ctxemb_file withctx.en-es.en.50.1.txt.ctx \
    --lang1_ctxemb_file withctx.en-es.es.50.1.txt.ctx \
    --lang01_desc_file es_wiktionary_final/10/es_en_train500_10.csv \
    --lang10_desc_file es_wiktionary_final/10/en_es_train500_10.csv \
    --mono_max_lines -1 \
    --multi_max_lines -1 \
    --model_root "$MODEL_ROOT" \
    --lang0_mono_index_corpus_file en_mono \
    --lang1_mono_index_corpus_file es_mono \
    --lang0_multi_index_corpus_file en_multi \
    --lang1_multi_index_corpus_file es_multi \
    --emb_dim 50 \
    --word2vec_batch_size 100000 \
    --bilbowa_sent_length 50 \
    --bilbowa_batch_size 100 \
    --encoder_desc_length 15 \
    --encoder_batch_size 64 \
    --encoder_lr 0.0002 \
    --train_mono=true \
    --train_multi=true \
    --train_encoder=true \
    --max_encoder_epochs 1000 \
    --timing_scaling_mono 3.0 \
    --timing_scaling_multi 3.0 \
    --word_emb_trainable=true \
    --context_emb_trainable=true \
    --encoder_target_no_gradient=true \
    --encoder_arch_version=1 \
    ;
}

mkdir -p "${__dir}/../scratch/model"

( cd "${__dir}/../code" && $1 )

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
  ./joint_prep.py \
    --data_root ../scratch/data \
    --lang0_emb_file withctx.en-fr.en.50.1.txt \
    --lang0_mono_string_corpus_file en_wiki_text_lower.txt \
    --lang0_multi_string_corpus_file europarl-v7.fr-en.en.tknzd.lower \
    --lang1_emb_file withctx.en-fr.fr.50.1.txt \
    --lang1_mono_string_corpus_file fr_wiki_text_lower.txt \
    --lang1_multi_string_corpus_file europarl-v7.fr-en.fr.tknzd.lower \
    --mono_max_lines -1 \
    --multi_max_lines -1 \
    --model_root ../scratch/model/joint_en_fr/ \
    --lang0_mono_index_corpus_file en_mono \
    --lang1_mono_index_corpus_file fr_mono \
    --lang0_multi_index_corpus_file en_multi \
    --lang1_multi_index_corpus_file fr_multi \
    ;
}

function en_es (){
  ./joint_prep.py \
    --data_root ../scratch/data \
    --lang0_emb_file withctx.en-es.en.50.1.txt \
    --lang0_mono_string_corpus_file en_wiki_text_lower.txt \
    --lang0_multi_string_corpus_file europarl-v7.es-en.en.tknzd.lower \
    --lang1_emb_file withctx.en-es.es.50.1.txt \
    --lang1_mono_string_corpus_file es_wiki_text_lower.txt \
    --lang1_multi_string_corpus_file europarl-v7.es-en.es.tknzd.lower \
    --mono_max_lines -1 \
    --multi_max_lines -1 \
    --model_root ../scratch/model/joint_en_es/ \
    --lang0_mono_index_corpus_file en_mono \
    --lang1_mono_index_corpus_file es_mono \
    --lang0_multi_index_corpus_file en_multi \
    --lang1_multi_index_corpus_file es_multi \
    ;
}


mkdir -p "${__dir}/../scratch/model"

( cd "${__dir}/../code" && $1 )

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

  ./joint_test.py \
    --data_root ../scratch/data \
    --lang0_emb_file withctx.en-fr.en.50.1.txt \
    --lang1_emb_file withctx.en-fr.fr.50.1.txt \
    --lang01_desc_file wiktionary_final/10/fr_en_train500_10.csv \
    --lang10_desc_file wiktionary_final/10/en_fr_train500_10.csv \
    --lang01_desc_test_file wiktionary_final/10/fr_en_test500.csv \
    --lang10_desc_test_file wiktionary_final/10/en_fr_test500.csv \
    --lang01_paraphrase_train_file para_en_fr_train75.csv \
    --lang01_paraphrase_test_file para_en_fr_test25.csv \
    --model_root "$MODEL_ROOT" \
    --word2vec_batch_size=1000000 \
    --encoder_batch_size=100 \
    ;
}

function en_es (){
  MODEL_ROOT=../scratch/model/joint_en_es/

  ./joint_test.py \
    --data_root ../scratch/data \
    --lang0_emb_file withctx.en-es.en.50.1.txt \
    --lang1_emb_file withctx.en-es.es.50.1.txt \
    --lang01_desc_file es_wiktionary_final/10/es_en_train500_10.csv \
    --lang10_desc_file es_wiktionary_final/10/en_es_train500_10.csv \
    --lang01_desc_test_file es_wiktionary_final/10/es_en_test500.csv \
    --lang10_desc_test_file es_wiktionary_final/10/en_es_test500.csv \
    --lang01_paraphrase_train_file para_en_es_train75.csv \
    --lang01_paraphrase_test_file para_en_es_test25.csv \
    --model_root "$MODEL_ROOT" \
    --word2vec_batch_size=1000000 \
    --encoder_batch_size=100 \
    ;
}

mkdir -p "${__dir}/../scratch/model"

# excute all functions if instructed
if [ "$1" = "all" ] ; then
  for func_name in `compgen -A function ` ; do
    echo "running \"$func_name\""
    ( cd "${__dir}/../code" && $func_name )
  done

  exit 0
fi

( cd "${__dir}/../code" && $1 )

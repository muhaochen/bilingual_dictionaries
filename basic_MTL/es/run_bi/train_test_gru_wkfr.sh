CUDA_VISIBLE_DEVICES=5 python gru_train_en_fr_bi.py en es ../../../data/es_wiktionary_final/10/es_en_train500_10.csv ../../../data/es_wiktionary_final/10/en_es_train500_10.csv ../model_bi/es_en_gru10_wk.h5

# 10
CUDA_VISIBLE_DEVICES=5 python tester.py 15 en es ../../../data/es_wiktionary_final/10/es_en_test500.csv ../model_bi/es_en_gru10_wk.h5 ../results_bi/es_en_gru10.txt ../../../data/es_wiktionary_final/10/es_en_train500_10.csv ../../../data/es_wiktionary_final/10/es_en_test500.csv ../../../data/es_wiktionary_final/10/en_es_train500_10.csv ../../../data/es_wiktionary_final/10/en_es_test500.csv
CUDA_VISIBLE_DEVICES=5 python tester.py 15 es en ../../../data/es_wiktionary_final/10/en_es_test500.csv ../model_bi/es_en_gru10_wk.h5 ../results_bi/en_es_gru10.txt ../../../data/es_wiktionary_final/10/en_es_train500_10.csv ../../../data/es_wiktionary_final/10/en_es_test500.csv ../../../data/es_wiktionary_final/10/es_en_train500_10.csv ../../../data/es_wiktionary_final/10/es_en_test500.csv

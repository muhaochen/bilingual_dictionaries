python gru_train_en_fr.py en es ../../../data/es_wiktionary_final/10/es_en_train500_10.csv ../model_mono/es_en_gru10_wk.h5
python gru_train_en_fr.py es en ../../../data/es_wiktionary_final/10/en_es_train500_10.csv ../model_mono/en_es_gru10_wk.h5


python tester.py 15 en es ../../../data/es_wiktionary_final/10/es_en_test500.csv ../model_mono/es_en_gru10_wk.h5 ../results_mono/es_en_gru10.txt ../../../data/es_wiktionary_final/10/es_en_train500_10.csv ../../../data/es_wiktionary_final/10/es_en_test500.csv ../../../data/es_wiktionary_final/10/en_es_train500_10.csv ../../../data/es_wiktionary_final/10/en_es_test500.csv
python tester.py 15 es en ../../../data/es_wiktionary_final/10/en_es_test500.csv ../model_mono/en_es_gru10_wk.h5 ../results_mono/en_es_gru10.txt ../../../data/es_wiktionary_final/10/en_es_train500_10.csv ../../../data/es_wiktionary_final/10/en_es_test500.csv ../../../data/es_wiktionary_final/10/es_en_train500_10.csv ../../../data/es_wiktionary_final/10/es_en_test500.csv

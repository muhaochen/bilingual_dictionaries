CUDA_VISIBLE_DEVICES=2 python att_train_en_fr.py en fr ../../../data/wiktionary_final/10/fr_en_train500_10.csv ../model/fr_en_att10_wk.h5
CUDA_VISIBLE_DEVICES=2 python att_train_en_fr.py fr en ../../../data/wiktionary_final/10/en_fr_train500_10.csv ../model/en_fr_att10_wk.h5


CUDA_VISIBLE_DEVICES=2 python tester.py 15 en fr ../../../data/wiktionary_final/10/fr_en_test500.csv ../model/fr_en_att10_wk.h5 ../results/fr_en_att10.txt ../../../data/wiktionary_final/10/fr_en_train500_10.csv ../../../data/wiktionary_final/10/fr_en_test500.csv ../../../data/wiktionary_final/10/en_fr_train500_10.csv ../../../data/wiktionary_final/10/en_fr_test500.csv
CUDA_VISIBLE_DEVICES=2 python tester.py 15 fr en ../../../data/wiktionary_final/10/en_fr_test500.csv ../model/en_fr_att10_wk.h5 ../results/en_fr_att10.txt ../../../data/wiktionary_final/10/en_fr_train500_10.csv ../../../data/wiktionary_final/10/en_fr_test500.csv ../../../data/wiktionary_final/10/fr_en_train500_10.csv ../../../data/wiktionary_final/10/fr_en_test500.csv

# Learning to Represent Bilingual Dictionaries (Lexical Definitions)

This is the repository for BilDRL: [Learning to Represent Bilingual Dictionaries](https://www.aclweb.org/anthology/K19-1015.pdf). This repository contains the source code and links to some datasets used in our paper. Particularly for the datasets, in addition to *bilingual lexical definitions* extracted from Wiktionary between English-French and English-Spanish, we also provide a cleaned set of *monolingual lexical definitions* in English, French and Spanish, also extracted from Wiktionary.

We propose a neural embedding model that learns to represent bilingual dictionaries (of lexical definitions). The proposed model is trained to map the lexical definitions to the cross-lingual target words, for which we explore with different sentence encoding techniques. To enhance the learning process on limited resources, our model adopts several critical learning strategies, including multi-task learning on different bridges of languages, and joint learning of the dictionary model with a bilingual word em-bedding model. We conduct experiments on two new tasks: cross-lingual reverse dictionary retrieval and bilingual paraphrase identification.

Environment:

    python 2.7 or 3.6
    Tensorflow >= 1.7 (with GPU support)
    Keras >= 2.2.4
    CuDNN

### Folders

- `./preprocess` contains preprocessing scripts to extract lexical definitions from raw Wiktionary dump of three languages, including a shell script for the entire process.  
- `./basic_MTL` contains basic and MTL model variants.  
- `./joint` contains the implementation of the joint training model,
    -  Here it is expected to execute `run_joint_prep.sh` before `run_joint_train.sh`.

These folders contain *shell scripts* to train/test. Please adjust paths to diretories for data/model accordingly if you would like to download and use the datasets and pre-tained embeddings (see below) we provide.

### Datasets and pre-trained embeddings

The link to the dataset can be found at [here](https://drive.google.com/drive/u/1/folders/1Lm6Q5BxeU0ByR6DZcNfbWpntumiIKhYN). 
Note that **all gzipped files need to be decompressed** after downloading, which, for example can be done by running `gzip -d *.gz` on a Linux box.

#### About the Wikt3l data files:  
In each csv file there are three columns. The first column is the translations of the target word in the source language, which are used just for reference to mask out surface information in the definitions and the monolingual baselines. Please consider the second column as label. The third column records the definitions.   

#### Monolingual lexical definitions:
We also include larger sets of monolingual lexical definitions for the above three languages. 

#### Other data files:
- `{en,es,fr}_wiki_text_lower.txt`: Processed monolingual corpora from Wikipedia.
- ` europarl-v7_*`: Processed multilingual corpora from Europarl Parallel Corpus.
- ` withctx.*`: Pre-trained multilingual word embeddings.

### Learning Bilingual Word Embeddings Using Lexical Definitions
Both the released monolingual and bilingual lexical defitions are used in our side project [Learning Bilingual Word Embeddings Using Lexical Definitions](https://www.aclweb.org/anthology/W19-4316/) described below.

In this side project, we use lexical definitions for bilingual word embedding learning. The BilLex model comprises a word pairing strategy to automatically identify and propagate the precise fine-grained word alignment from lexical definitions. We evaluate BilLex in word-level and sentence-level translation tasks. More details of this side project can be found at [here](https://github.com/swj0419/bilingual_dict_embeddings).

## Reference
If you find our resources useful to you, please kindly cite our paper.  
Bibtex:

    @inproceedings{chen2019bildict,
        title={Learning to Represent Bilingual Dictionaries},
        author={Chen, Muhao and Tian, Yingtao and Chen, Haochen and Chang, Kai-Wei and Skiena, Steven and Zaniolo, Carlo},
        booktitle={The 23rd SIGNLL Conference on Computational Natural Language Learning (CoNLL)},
        year = {2019},
        publisher={ACL}
    }

# Representation Learning for Bilingual Lexical Definitions

This is the repository for BilDRL: Learning to Represent Bilingual Dictionaries. This repository contains the source code and links to some datasets used in our paper. Particularly for the datasets, in addition to *bilingual lexical definitions* extracted from Wiktionary between English-French and English-Spanish, we also provide a cleaned set of *monolingual lexical definitions* in English, French and Spanish, also extracted from Wiktionary.

## Learning to Represent Bilingual Dictionaries

Bilingual word embeddings have been widely used to capture the correspondence of lexical semantics in different human languages. However, the cross-lingual correspondencebetween sentences and words is less studied, despite that this correspondence can significantly benefit many applications such as cross-lingual semantic search and textual inference.To bridge this gap, we propose a neural embedding model that leverages bilingual dictionaries1. The proposed model is trained to mapthe lexical definitions to the cross-lingual target words, for which we explore with different sentence encoding techniques. To enhance the learning process on limited resources, our model adopts several critical learning strategies, including multi-task learning on different bridges of languages, and joint learning of the dictionary model with a bilingual word em-bedding model. We conduct experiments on two new tasks: cross-lingual reverse dictionary retrieval and bilingual paraphrase identification.

Environment:

    python 2.7 or 3.6
    Tensorflow 1.7 (with GPU support)
    CuDNN
    Keras 2.2.4

### Folders

./basic_MTL contains basic and MTL model variants.  
./joint contains the implementation of the joint training model.  
These folders contain shell scripts to train/test.  
./preprocess contains preprocessing scripts to extract lexical definitions from raw Wiktionary dump of three languages, including a shell script for the entire process.  


### Datasets and pre-trained embeddings

The link to the dataset can be found at [here](https://drive.google.com/drive/u/1/folders/1Lm6Q5BxeU0ByR6DZcNfbWpntumiIKhYN).  
#### About the Wikt3l data files:  
In each csv file there are three columns. The first column is the translations of the target word in the source language, which are used just for reference to mask out surface information in the definitions and the monolingual baselines. Please consider the second column as label. The third column records the definitions.   
#### Monolingual lexical definitions:
We also include larger sets of monolingual lexical definitions for the above three languages. Together with the bilingual lexical defitions, they are used in paper [Learning Bilingual Word Embeddings Using Lexical Definitions](https://www.aclweb.org/anthology/W19-4316/).

## Learning Bilingual Word Embeddings Using Lexical Definitions
we propose BilLex that leverages publicly available lexical definitions for bilingual word embedding learning. Without the need of predefined seed lexicons, BilLex comprises a novel word pairing strategy to automatically identify and propagate the precise finegrained word alignment from lexical definitions. We evaluate BilLex in word-level and sentence-level translation tasks, which seek to find the cross-lingual counterparts of words and sentences respectively. (To be updated)

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

-

    @inproceedings{shi2019bilex,
        title={Learning Bilingual Word Embeddings Using Lexical Definitions},
        author={Shi, Weijia and Chen, Muhao and Tian, Yingtao and Chang, Kai-Wei},
        booktitle={RepL4NLP},
        year = {2019},
        publisher={ACL}
    }

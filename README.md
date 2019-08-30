## Learning to Represent Bilingual Dictionaries

This is the repository for BilDRL. This repository contains the source code and links to some datasets used in our paper.

Environment:

    python 2.7 or 3.6
    Tensorflow 1.7 (with GPU support)
    CuDNN
    Keras 2.2.4

## Folders

./basic_MTL contains basic and MTL model variants.  
./joint contains the implementation of the joint training model.  
These folders contain shell scripts to train/test.  
./preprocess contains preprocessing scripts to extract lexical definitions from raw Wiktionary dump of three languages, including a shell script for the entire process.  


## Datasets and pre-trained embeddings

The link to the dataset can be found at (here)[https://drive.google.com/drive/u/1/folders/1Lm6Q5BxeU0ByR6DZcNfbWpntumiIKhYN].  
### About the Wikt3l data files:  
In each csv file there are three columns. The first column is the translations of the target word in the source language, which are used just for reference to mask out surface information in the definitions and the monolingual baselines. Please consider the second column as label. The third column records the definitions.   

## Reference

Bibtex:

    @inproceedings{chen2019bildict,
        title={Learning to Represent Bilingual Dictionaries},
        author={Chen, Muhao and Tian, Yingtao and Chen, Haochen and Chang, Kai-Wei and Skiena, Steven and Zaniolo, Carlo},
        booktitle={The 23rd SIGNLL Conference on Computational Natural Language Learning (CoNLL)},
        year = {2019},
        publisher={ACL}
    }

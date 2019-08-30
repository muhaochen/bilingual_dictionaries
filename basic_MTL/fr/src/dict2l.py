
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import csv
import scipy
import heapq as HP
import tqdm

def safe_lower(word):
    x = unicode(word, 'utf-8')
    x = x.lower()
    return x.encode('utf-8')

class dict2l(object):

    def __init__(self):
        
        # wv, mapping from lan to object
        self.lan = set([])
        self.tokens = {}
        self.wv = {}
        self.token_index = {}
        self.n_tokens = {}
        
        # descriptions
        self.desc = {}
        self.desc_embed = {}
        self.valid_length = {}
        # (desc_index, word_index) pairs
        self.dw_pair = {}
        self.w2d = {}
        self.d2w = {}
        self.length = 10
        self.cand_vocab = {}
        
        self.dim = -1
    
    def load_stopwords(self, filepath):
        stop_words = []
        for line in open(filepath):
            stop_words.append(line.strip())
        return set(stop_words)
    
    def load_word2vec(self, filepath, stop_words=set([]), lan = 'en', splitter=' ', filt_stop=False):
        self.lan.add(lan)
        self.tokens[lan], emb = [], []
        for lineno, l in tqdm.tqdm(enumerate(open(filepath)), desc='load word embedding', unit=' word'):
            tokens = l.strip().split(splitter)
            if lineno == 0:
                if self.dim > 0:
                    dim = int(tokens[1])
                    assert(dim == self.dim)
                else:
                    self.dim = dim = int(tokens[1])
                emb.append(np.zeros(dim))
                self.tokens[lan].append('  ')
                continue
            if len(tokens) == 1 + dim and ((not filt_stop) or tokens[0] not in stop_words):
                self.tokens[lan].append(tokens[0])
                emb.append([float(_) for _ in tokens[1:]])
        self.wv[lan] = np.array(emb)
        self.token_index[lan] = {w:i for i, w in enumerate(self.tokens[lan])}
        self.n_tokens[lan] = len(self.tokens)
        self.valid_length[lan] = []
        print("Loaded token embeddings from", filepath, '. Totally', len(self.wv[lan]), 'words')

    def load_word2vec_vocab(self, filepath, vocab=set([]), lan = 'en', splitter=' '):
        self.lan.add(lan)
        self.tokens[lan], emb = [], []
        for lineno, l in tqdm.tqdm(enumerate(open(filepath)), desc='load word embedding', unit=' word'):
            tokens = l.strip().split(splitter)
            if lineno == 0:
                if self.dim > 0:
                    dim = int(tokens[1])
                    assert(dim == self.dim)
                else:
                    self.dim = dim = int(tokens[1])
                emb.append(np.zeros(dim))
                self.tokens[lan].append('  ')
                continue
            if tokens[0] in vocab and len(tokens) == 1 + dim:
                self.tokens[lan].append(tokens[0])
                emb.append([float(_) for _ in tokens[1:]])
        self.wv[lan] = np.array(emb)
        self.token_index[lan] = {w:i for i, w in enumerate(self.tokens[lan])}
        self.n_tokens[lan] = len(self.tokens)
        self.valid_length[lan] = []
        print("Loaded token embeddings from", filepath)
    
    def w2v(self, word, lan='en'):
        assert(lan in self.lan)
        tk = self.token_index[lan].get(word)
        if tk is not None:
            return self.i2v(tk, lan)
    
    def i2v(self, index, lan='en'):
        assert(lan in self.lan)
        return self.wv[lan][index]

    def set_length(self, len):
        assert(len > 0)
        self.length = len
    
    def embed_desc(self, desc, lan='en', lower=True, zero_pad = True, record_valid_length = False, remove_oov=True):
        assert(lan in self.lan)
        tks = desc.split(' ')
        embed = []
        hit = False
        vl = 0
        for t in tks:
            if lower:
                t = safe_lower(t)
            v = self.w2v(t, lan)
            if v is None:
                if not remove_oov:
                    v = np.zeros(self.dim)
                else:
                    continue
            elif hit == False:
                hit = True
            embed.append(v)
            if (not record_valid_length) and len(embed) == self.length:
                break
        vl = len(embed)
        if not hit:
            return None
        if record_valid_length and len(embed) > self.length:
            embed = embed[:self.length]
        if zero_pad:
            while len(embed) < self.length:
                embed.append(np.zeros(self.dim))
        else:
            original_embed = list(embed)
            while len(embed) < self.length:
                embed.extend(original_embed)
            if len(embed) > self.length:
                embed = embed[:self.length]
        if record_valid_length:
            self.valid_length[lan].append(vl)
        return np.array(embed)

    def embed_desc_index(self, desc, lan='en', zero_pad = True, record_valid_length = False):
        assert(lan in self.lan)
        tks = desc
        embed = []
        hit = False
        vl = 0
        for t in tks:
            v = self.i2v(t, lan)
            if v is None:
                v = np.zeros(self.dim)
            elif hit == False:
                hit = True
            embed.append(v)
            if (not record_valid_length) and len(embed) == self.length:
                break
        vl = len(embed)
        if not hit:
            return None
        if record_valid_length and len(embed) > self.length:
            embed = embed[:self.length]
        if zero_pad:
            while len(embed) < self.length:
                embed.append(np.zeros(self.dim))
        else:
            original_embed = list(embed)
            while len(embed) < self.length:
                embed.extend(original_embed)
            if len(embed) > self.length:
                embed = embed[:self.length]
        if record_valid_length:
            self.valid_length[lan].append(vl)
        return np.array(embed)
    
    # need pool_name (such as en2fr_train)
    def word_desc_pool(self, filepath, pool_name, lower=True, src_lan = 'en', tgt_lan = 'fr', splitter=',', save_desc=True, min_len=-1):
        assert(src_lan in self.lan and tgt_lan in self.lan)
        self.w2d[pool_name], self.d2w[pool_name] = {}, {}
        self.dw_pair[pool_name], self.desc_embed[pool_name] = [], []
        if save_desc:
            self.desc[pool_name] = []
        did = 0
        for line in tqdm.tqdm(csv.reader(open(filepath), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)):
            word = line[-2]
            desc = line[-1]
            if len(word) == 0 or (min_len > 0 and len(word) < min_len):
                continue
            wid = self.token_index[tgt_lan].get(word)
            if wid is None:
                continue
            desc_embed = self.embed_desc(desc, src_lan, lower, True, True)
            if desc_embed is None:
                continue
            if self.w2d[pool_name].get(wid) is None:
                self.w2d[pool_name][wid] = set([])
            self.w2d[pool_name][wid].add(did)
            self.d2w[pool_name][did] = wid
            self.dw_pair[pool_name].append((did, wid))
            self.desc_embed[pool_name].append(desc_embed)
            did += 1
            if save_desc:
                self.desc[pool_name].append(desc.split(' '))
        self.dw_pair[pool_name] = np.array(self.dw_pair[pool_name])
        self.desc_embed[pool_name] = np.array(self.desc_embed[pool_name])
        print("Finished loading descriptions and description embeddings <", pool_name ,"> from ", filepath, '(', len(self.desc_embed[pool_name]) ,')')
        if save_desc:
            print("Average length is", np.mean(self.valid_length[src_lan]))
    
    def load_vocab_set(self, filepath, lan1, lan2, lan2_only=False):
        if self.cand_vocab.get(lan1) is None:
            self.cand_vocab[lan1] = set([])
        if self.cand_vocab.get(lan2) is None:
            self.cand_vocab[lan2] = set([])
        for line in tqdm.tqdm(csv.reader(open(filepath), delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)):
            word = line[-2]
            desc = line[-1]
            if len(word) == 0:
                continue
            wid = self.token_index[lan2].get(word)
            if wid is None:
                continue
            self.cand_vocab[lan2].add(wid)
            if lan2_only:
                continue
            for w in desc.split(' '):
                if len(w) < 2:
                    continue
                tid = self.token_index[lan1].get(w)
                if tid is not None:
                    self.cand_vocab[lan1].add(tid)
        print("Extended candidate vocab from",filepath)
        print(lan1,":",len(self.cand_vocab[lan1]),'. ',lan2, ":", len(self.cand_vocab[lan2]))

    def clear_vocab_set(self):
        self.cand_vocab = {}

    
    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return
        def __lt__(self, other):
            return self.dist > other.dist

    def distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)
        #return scipy.spatial.distance.cosine(v1, v2)
    
    def kNN(self, vec, lan, topk=10, self_id=None, except_ids=None, limit_ids=None):
        assert(lan in self.lan)
        q = []
        cand_ids = limit_ids
        if cand_ids is None:
            cand_ids = range(len(self.wv[lan]))
        for i in cand_ids:
            #skip self
            if i == self_id or ((not except_ids is None) and i in except_ids):
                continue
            #if (not limit_ids is None) and i not in limit_ids:
                #continue
            dist = self.distance(vec, self.wv[lan][i])
            if len(q) < topk:
                HP.heappush(q, self.index_dist(i, dist))
            else:
                #indeed it fetches the biggest
                tmp = HP.nsmallest(1, q)[0]
                if tmp.dist > dist:
                    HP.heapreplace(q, self.index_dist(i, dist) )
        rst = []
        while len(q) > 0:
            item = HP.heappop(q)
            rst.insert(0, (item.index, item.dist))
        return rst
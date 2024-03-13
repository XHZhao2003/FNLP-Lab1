import random
import json
import numpy as np
from tqdm import tqdm

class Mydataset:
    def __init__(self, vocab_path, feat_path, data_path, batchsize) -> None:
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        with open(feat_path, 'r') as f:
            self.featlist = json.load(f)
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
    
        # shuffle the data
        shuffle_index = [x for x in range(len(self.data))]
        random.shuffle(shuffle_index)
        data = [(self.data[x][1] + self.data[x][2]) for x in shuffle_index]
        label = [(int)(self.data[x][0]) - 1 for x in shuffle_index]
        self.data = data
        self.label = label
        
        # calculate TDIDF feature
        feats = []
        
        idf_path = '../data/unigram_idf.json'
        with open(idf_path, 'r') as f:
            idf = json.load(f)
        
        print("Calculate TFIDF-feature for dataset")
        for sample in tqdm(self.data, total=len(self.data)):
            tf = {x : 0 for x in self.featlist}
            for word in sample:
                if word in tf:
                    tf[word] += 1
            for word in tf:
                if tf[word] > 0:
                    tf[word] = 1 + np.log10(tf[word])
                        
            tfidf = [tf[x] * idf[x] for x in tf]
            feats.append(tfidf)
        
        self.data = np.array(feats)
            
        
        self.cur = 0
        self.batchsize = batchsize
        
    def __getitem__(self, index):
        if self.cur + self.batchsize <= len(self.data):
            data = self.data[self.cur : self.cur + self.batchsize]
            label = self.label[self.cur : self.cur + self.batchsize]
            self.cur += self.batchsize
            return data, label
        elif self.cur < len(self.data):
            data = self.data[self.cur : len(self.data)]
            label = self.label[self.cur : len(self.data)]
            self.cur = len(self.data)
            return data, label
        else:
            self.cur = 0
            raise StopIteration()
    
    def __len__(self):
        res = len(self.data) // self.batchsize
        if len(self.data) % self.batchsize > 0:
            res += 1
        return res
    
# vocab_path = '../data/vocab.json'
# feat_path = '../data/unigram_list.json'
# data_path = '../data/test_cleaned.json'

# train_dataset = Mydataset(vocab_path=vocab_path, feat_path=feat_path, 
#                           data_path=data_path, batchsize=512)
# for feat, label in train_dataset:
#     print(feat[0])
#     break
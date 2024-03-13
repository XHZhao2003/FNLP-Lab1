import json
from nltk.corpus import stopwords


vocabulary_filepath = '../../data/vocab.json'
dst_path = '../../data/unigram_list.json'
unigram_frequency = 500     # 选取多于500次出现的unigram作为特征

with open(vocabulary_filepath, 'r') as f:
    vocab = json.load(f)
stoplist = stopwords.words('english')
    
unigram_list = []
for word in vocab:
    if word not in stoplist and vocab[word] >= unigram_frequency:
        unigram_list.append(word)

with open(dst_path, 'w') as f:
    json.dump(unigram_list, f, indent = 4)
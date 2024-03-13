import json
import numpy as np
from tqdm import tqdm

featlist_path = '../../data/unigram_list.json'
train_path = '../../data/train_cleaned.json'
dst_path = '../../data/unigram_idf.json'

with open(featlist_path, 'r') as f:
    featlist = json.load(f)
with open(train_path, 'r') as f:
    train_data = json.load(f)
    
df = {x : 0 for x in featlist}
for documment in tqdm(train_data, total=len(train_data)):
    doc = documment[1] + documment[2]
    for x in df:
        if x in doc:
            df[x] += 1
            
N = len(train_data)
idf = {x : np.log10(N / df[x]) for x in df}

with open(dst_path, 'w') as f:
    json.dump(idf, f, indent=4)

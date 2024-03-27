import json
import random

src_file = '../../data/train_cleaned.json'
dst_train = '../../data/train.json'
dst_valid = '../../data/valid.json'

valid_length = 7600

with open(src_file, 'r') as f:
    train_cleaned = json.load(f)

train_length = len(train_cleaned) - valid_length
index = [x for x in range(len(train_cleaned))]
random.shuffle(index)

with open(dst_valid, 'w') as f:
    valid = [train_cleaned[i] for i in index[:valid_length]]
    json.dump(valid, f, indent=4)

with open(dst_train, 'w') as f:
    train = [train_cleaned[i] for i in index[valid_length:]]
    json.dump(train, f, indent=4)
    
    
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import csv
import json


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# A sentence -> a list of lemmatized tokens
def lemmatize(content):
    pos_tags = pos_tag(word_tokenize(content))
    wnl = WordNetLemmatizer()
    lemmas = []
    for tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
    return lemmas

def remove_nonletter(content):
    content = content.replace('-', ' ')
    content = content.replace('/', ' ')
    content = content.replace('\\', ' ')
    content = content.replace('\\\\', ' ')
    return content


train_filepath = '../../data/train.csv'
test_filepath = '../../data/test.csv'
dst_train = '../../data/train_cleaned.json'
dst_test = '../../data/test_cleaned.json'
dst_vocabulary = '../../data/vocab.json'

train_lemma = []
test_lemma = []

# lemmatization for train and test
with open(train_filepath, 'r') as f:
    train_data = list(csv.reader(f))
    for line in tqdm(train_data, total=len(train_data)): 
        title = lemmatize(remove_nonletter(line[1]))
        description = lemmatize(remove_nonletter(line[2]))
        
        title = [x.lower() for x in title if x.isalpha()]
        description = [x.lower() for x in description if x.isalpha()]
                
        train_lemma.append([line[0], title, description])
        
with open(test_filepath, 'r') as f:
    test_data = list(csv.reader(f))
    for line in tqdm(test_data, total=len(test_data)):
        title = lemmatize(remove_nonletter(line[1]))
        description = lemmatize(remove_nonletter(line[2]))
        
        title = [x.lower() for x in title if x.isalpha()]
        description = [x.lower() for x in description if x.isalpha()]

        test_lemma.append([line[0], title, description])

# collecting vocabulary list
vocabulary = {}
for line in tqdm(train_lemma, total=len(train_lemma)):
    content = line[1] + line[2]
    for word in content:
        if not word.isalpha():
            continue
        
        word = word.lower()
        if word not in vocabulary:
            vocabulary[word] = 1
        else:
            vocabulary[word] += 1

with open(dst_train, 'w') as f:
    json.dump(train_lemma, f, indent=4)
with open(dst_test, 'w') as f:
    json.dump(test_lemma, f, indent=4)
with open(dst_vocabulary, 'w') as f:
    json.dump(vocabulary, f, indent=4)    



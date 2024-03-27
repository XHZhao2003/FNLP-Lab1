from model import LogLinearModel
from dataset import Mydataset
from tqdm import tqdm
from utils import metric, metric_by_sklearn
import sys

import numpy as np

test_path = '../data/test.json'
vocab_path = '../data/vocab.json'
feat_path = '../data/unigram_list.json'
model_path = '../model/model.txt'

model_path = '../model/model' +  sys.argv[1] + '.txt'

label_dim = 4

test_dataset = Mydataset(vocab_path=vocab_path, feat_path=feat_path,
                         data_path=test_path, batchsize=32)
model = LogLinearModel()
model.load(model_path)

# Test performance of model
true_positive = [0 for i in range(label_dim)]
false_negative = [0 for i in range(label_dim)]
false_positive = [0 for i in range(label_dim)]

total_pred_label = []
total_true_label = []

for feat, label in test_dataset:
    pred = model.predict(feat)
    pred_label = list(np.argmax(pred, axis=1))
    total_pred_label += pred_label
    total_true_label += label
        
    for i in range(len(pred_label)):
        if pred_label[i] == label[i]:
            true_positive[label[i]] += 1
        else:
            false_positive[pred_label[i]] += 1
            false_negative[label[i]] += 1
            
accuracy, f1 = metric(true_positive, false_positive, false_negative, label_dim)
print("Testing Metric: Accrucy = %.3f, F1 = %.3f" % (accuracy, f1))

print("Metric by sklearn:")
print(metric_by_sklearn(total_pred_label, total_true_label))

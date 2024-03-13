from model import LogLinearModel
from dataset import Mydataset
from tqdm import tqdm
from utils import metric, metric_by_sklearn

import numpy as np

train_path = '../data/train_cleaned.json'
test_path = '../data/test_cleaned.json'
vocab_path = '../data/vocab.json'
feat_path = '../data/unigram_list.json'


feat_dim = 1024
label_dim = 4
batchsize = 64
lr_rate = 0.01
epoch_num = 15
beta = 0.1          # L2正则化系数

print("Batchsize = %d, alpha = %f, beta = %f, epoch_num = %d" % (batchsize, lr_rate, beta, epoch_num))

train_dataset = Mydataset(vocab_path=vocab_path, feat_path=feat_path, 
                          data_path=train_path, batchsize=batchsize)
test_dataset = Mydataset(vocab_path=vocab_path, feat_path=feat_path,
                         data_path=test_path, batchsize=batchsize)
model = LogLinearModel(feat_dim=feat_dim, label_dim=label_dim, alpha=lr_rate, beta=beta)

for i in tqdm(range(epoch_num)):
    loss = 0
    for feat, label in tqdm(train_dataset, total=len(train_dataset)):
        loss += model.update(feat, label)
    loss = loss / len(train_dataset)
        
    print("Epoch %d: training loss = %f" % (i + 1, loss))
    

# Test performance of model
true_positive = [0 for i in range(label_dim)]
false_negative = [0 for i in range(label_dim)]
false_positive = [0 for i in range(label_dim)]

total_pred_label = []
total_true_label = []

for feat, label in tqdm(test_dataset):
    pred = model.predict(feat)
    pred_label = list(np.argmax(pred, axis=1))
    total_pred_label += pred_label
    total_true_label += label
    
    assert len(pred_label) == len(label)
    
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

from model import LogLinearModel
from dataset import Mydataset
from tqdm import tqdm
from utils import metric, metric_by_sklearn
import sys

import numpy as np

train_path = '../data/train.json'
valid_path = '../data/valid.json'
vocab_path = '../data/vocab.json'
feat_path = '../data/unigram_list.json'
model_path = '../model/model.txt'
model_path_prefix = '../model/model'


feat_dim = 1024
label_dim = 4
batchsize = 32
lr_rate = 0.01
epoch_num = 50
beta = 0.001          # L2正则化系数
gamma = 0.99

if len(sys.argv) > 1:
    lr_rate = float(sys.argv[1])
    beta = float(sys.argv[2])
    gamma = float(sys.argv[3])
    model_path = model_path_prefix + sys.argv[4] + '.txt'

print("Batchsize = %d, alpha = %f, beta = %f, epoch_num = %d, gamma = %f" % (batchsize, lr_rate, beta, epoch_num, gamma))

accuracy_list = []
f1_list = []

train_dataset = Mydataset(vocab_path=vocab_path, feat_path=feat_path, 
                          data_path=train_path, batchsize=batchsize)
model = LogLinearModel(feat_dim=feat_dim, label_dim=label_dim, alpha=lr_rate, beta=beta, weight_decay=gamma)

for i in tqdm(range(epoch_num)):
    loss = 0
    reg_loss = 0
    for feat, label in train_dataset:
        batch_loss, batch_reg_loss = model.update(feat, label)
        loss += batch_loss
        reg_loss += batch_reg_loss
    loss = loss / len(train_dataset)
    reg_loss  = reg_loss / len(train_dataset)
    
    model.decayed_lr = model.decayed_lr * model.gamma
    print("Epoch %d: training loss = %f, reg loss = %f" % (i + 1, loss, reg_loss))
    sys.stdout.flush()

model.save(model_path)

valid_dataset = Mydataset(vocab_path=vocab_path, feat_path=feat_path, 
                          data_path=valid_path, batchsize=batchsize)
# Test performance of model
true_positive = [0 for i in range(label_dim)]
false_negative = [0 for i in range(label_dim)]
false_positive = [0 for i in range(label_dim)]

total_pred_label = []
total_true_label = []

for feat, label in valid_dataset:
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
    
print("Valid Metric: Accrucy = %.3f, F1 = %.3f" % (accuracy, f1))
    

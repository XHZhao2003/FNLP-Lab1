from sklearn import metrics
import numpy as np



def metric(true_positive, false_positive, false_negative, label_dim=4):
    total_num = sum(true_positive) + sum(false_positive)
    accuracy = sum(true_positive) / total_num
    
    precision = [(true_positive[i] / (true_positive[i] + false_positive[i])) for i in range(label_dim)]
    recall = [(true_positive[i] / (true_positive[i] + false_negative[i])) for i in range(label_dim)]
    f1 = [(2 * precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(label_dim)]
    macro_f1 = sum(f1) / label_dim
    
    return accuracy, macro_f1

def metric_by_sklearn(pred_label, true_label):
    return metrics.classification_report(pred_label, true_label, digits=3)


from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

import numpy as np
from sklearn.metrics import roc_curve

def ood_auc(label,socre):
    return roc_auc_score(label, socre)

def ood_aupr(label,score):
    return average_precision_score(label,score)

def fpr95(y_true, y_scores):
    # 计算ROC曲线的值
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # 寻找第一个tpr >= 0.95的索引
    target = np.where(tpr >= 0.95)[0][0]

    # 返回该点的FPR
    return fpr[target]

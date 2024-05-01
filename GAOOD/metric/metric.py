

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

def ood_auc(label,socre):
    return roc_auc_score(label, socre)

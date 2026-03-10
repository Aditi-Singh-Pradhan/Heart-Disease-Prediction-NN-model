import numpy as np


def accuracy_np(y_true, y_pred):
    return np.mean(np.asarray(y_true) == np.asarray(y_pred))


def confusion_matrix_np(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp],
                     [fn, tp]])


def precision_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-12)


def recall_np(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-12)


def f1_np(y_true, y_pred):
    p = precision_np(y_true, y_pred)
    r = recall_np(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-12)


def roc_curve_np(y_true, y_score):
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    desc    = np.argsort(-y_score)
    y_sorted = y_true[desc]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    if P == 0 or N == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.nan, np.nan])

    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    tpr = np.concatenate([[0.0], tp_cum / P, [1.0]])
    fpr = np.concatenate([[0.0], fp_cum / N, [1.0]])
    thresholds = np.concatenate([[np.inf], y_score[desc], [-np.inf]])
    return fpr, tpr, thresholds


def auc_from_roc(fpr, tpr):
    return np.trapz(tpr, fpr)


def print_metrics(y_true, y_pred, y_probs):
    acc  = accuracy_np(y_true, y_pred)
    prec = precision_np(y_true, y_pred)
    rec  = recall_np(y_true, y_pred)
    f1   = f1_np(y_true, y_pred)
    fpr, tpr, _ = roc_curve_np(y_true, y_probs)
    auc  = auc_from_roc(fpr, tpr)
    cm   = confusion_matrix_np(y_true, y_pred)

    print("\nTest metrics:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  AUC       : {auc:.4f}")
    print("Confusion matrix (rows: actual 0/1, cols: pred 0/1):\n", cm)
    return fpr, tpr, auc

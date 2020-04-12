import numpy as np
from sklearn.metrics import confusion_matrix

""" Found out this was allowed on Piazza. Just cant use for training and testing"""
BETA = 1


def evaluate(truth_labels, ybar):
    """ Evaluates a prediction compared to the ground truth labels according to a number of
    different metrics. At the moment, so it can be averaged across partitions, just returning fscore"""
    assert (len(truth_labels) == len(ybar))
    print("Accuracy: " + str(accuracy(truth_labels, ybar)))
    cm = confusion_matrix(truth_labels, ybar)
    p = precision(cm)
    r = recall(cm)
    print("Weighted Precision: " + str(p))
    print("Weighted Recall: " + str(r))
    print("F-score (Beta = " + str(BETA) + "): " + str(f_score(p, r, BETA)))
    return f_score(p, r, BETA)


def accuracy(class_col, ybar):
    results = np.array(class_col) == ybar
    return np.count_nonzero(results == True) / len(ybar)


def precision(cm):
    """Precision of each class, returned as an average weighted by the number of
    instances in each class"""
    fp = np.sum(cm, axis=0)
    precisions = np.diag(cm) / np.where(fp == 0, 1, fp)
    weights = np.sum(cm, axis=1) / cm.sum()
    return np.sum(precisions * weights)


def recall(cm):
    """Recall of each class, returned as an average weighted by the number of
    instances in each class"""
    recalls = np.diag(cm) / np.sum(cm, axis=1)
    weights = np.sum(cm, axis=1) / cm.sum()
    return np.sum(recalls * weights)


def f_score(p, r, beta):
    return ((1 + beta * beta) * p * r) / ((beta * beta * p) + r)

import src.niave_bayes as nb
import src.evaluate as ev
import numpy as np
import pandas as pd
from collections import defaultdict as dd


def k_split(total, k):
    """ basically calculated the indices by which to slice the dataset into k partitions"""
    quotient, remainder = divmod(total, k)
    indices = [quotient + 1] * remainder + [quotient] * (k - remainder)
    new = [indices[0]]
    for i in range(1, k - 1):
        new.append(new[i - 1] + indices[i])
    return new


def partition(df, k):
    """Splits the dataset into k partitions, and then allocates each in turn as the test set,
        whilst the remainder are used for training"""
    partition_lengths = k_split(df.shape[0], k)
    partitions = np.array_split(df, partition_lengths)
    splits = list()
    for i in range(k):
        train = pd.concat(partitions[:i] + partitions[i + 1:])
        test = partitions[i]
        splits.append((train, test))
    return splits


def cross_validation(df, cnfg, k):
    results = np.zeros((4, 1))
    for train, test in partition(df, k):
        model = nb.train(train, cnfg["discrete"], cnfg["numeric"], cnfg["class_col"])
        predictions = nb.predict(test, model)
        truth = test[cnfg["class_col"]]
        a, p, r, f = ev.evaluate(truth, predictions, print=False)
        results[0] += a
        results[1] += p
        results[2] += r
        results[3] += f
    return results / k

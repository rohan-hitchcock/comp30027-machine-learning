import src.niave_bayes as nb
import numpy as np
import pandas as pd

NUM_PARTITIONS = 10


def k_split(total, k):
    """ basically calculated the indices by which to slice the dataset into k partitions"""
    quotient, remainder = divmod(total, k)
    indices = [quotient + 1] * remainder + [quotient] * (k - remainder)
    new = [indices[0]]
    for i in range(1, k - 1):
        new.append(new[i - 1] + indices[i])
    return new


def cross_validation(data, k):
    """Splits the dataset into k partitions, and then allocates each in turn as the test set,
        whilst the remainder are used for training"""
    partition_lengths = k_split(data.shape[0], k)
    partitions = np.array_split(data, partition_lengths)
    splits = list()
    for i in range(k):
        train = pd.concat(partitions[:i] + partitions[i + 1:])
        test = partitions[i]
        splits.append((train, test))
    return splits
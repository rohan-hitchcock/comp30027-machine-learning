import src.niave_bayes as nb
import numpy as np
import pandas as pd

NUM_PARTITIONS = 10


def train(data, cnfg, conditional=nb.conditional_laplace):
    """ given a data set, train returns a list of 3 tuples including the niave bayers model,
        discrete priors, and the test set according to the cross validation partitions"""
    class_col = cnfg['class_col']
    models = list()
    for train_set, test_set in cross_validation(data, NUM_PARTITIONS):
        model = nb.calculate_conditionals_discrete(train_set, class_col, conditional)
        priors = nb.discrete_priors(train_set[class_col], np.unique(train_set[class_col]))
        models.append((model, priors, test_set))
    return models


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
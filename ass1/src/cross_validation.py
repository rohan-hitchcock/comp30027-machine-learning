
"""
import src.niave_bayes as nb
import src.evaluate as ev
"""

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
    
    print(type(partitions))
    
    splits = list()
    for i in range(k):
        train = pd.concat(partitions[:i] + partitions[i + 1:])
        test = partitions[i]
        splits.append((train, test))
    return splits

"""
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
"""


def cross_validation_splits(df, k, seed=0):

    #set seed for reproducibility
    np.random.seed(seed)

    indicies = df.index.to_numpy(copy=True)
    np.random.shuffle(indicies)

    partitions = np.array_split(indicies, k)
    for i in range(k):

        test = partitions[i]
        train = np.concatenate(partitions[:i] + partitions[i + 1:])

        yield df.loc[train], df.loc[test]
    

if __name__ == "__main__":
    df = pd.read_csv("../datasets/lymphography.data")

    df = df.head(20)
    print(df)

    i = 0
    for train, test in cross_validation_splits(df, 5):
        i += 1

        print(f"crossval split {i}")
        print("train:")
        print(train)
        print()
        print("test:")
        print(test)
        print()
        print()


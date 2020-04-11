import numpy as np


def zero_r(class_col):
    values, counts = np.unique(class_col, return_counts=True)
    ind = np.argmax(counts)
    return np.full(class_col.shape, values[ind])

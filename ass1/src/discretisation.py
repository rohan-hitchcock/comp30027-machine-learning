import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import KBinsDiscretizer as KBD


def discretise_equal_width(numeric_series, nbins):
    """ Discretises a numeric data series accoding to equal-width discretisation
        
        Assumes no missing values.

        Args:
            numeric_series: a pd.Series containing numeric data
            nbins: the number of discrete classes for the data
            
        Returns:
            A pd.Series with nbins unique values.
    """

    est = KBD(n_bins=nbins, encode='ordinal', strategy='uniform')
    est.fit(numeric_series)
    return pd.DataFrame(est.transform(numeric_series), dtype='int64', columns=numeric_series.columns)


def discretise_k_means(numeric_series, k):
    """ Discretises a numeric data series according to k-means 

        Assumes no missing values.

        Args:
            numeric_series: a pd.Series containing numeric data
            k: the number of discrete categories
            repeates: the number of repeats of k-means
        
        Returns:
            A pd.Series with k unique values of the same index as numeric_series
    """

    est = KBD(n_bins=k, encode='ordinal', strategy='kmeans')
    est.fit(numeric_series)
    return pd.DataFrame(est.transform(numeric_series), dtype='int64', columns=numeric_series.columns)


def mode(xs):
    """ Returns one mode of a sequence xs.

        Args:
            xs: a seqence 

        Returns:
            one of the modes of xs, ignoring NaN values
    """
    vals, counts = np.unique(xs, return_counts=True)

    # find all indices for which the maxium count occurs
    modes_indexes = np.argwhere(counts == np.nanmax(counts)).flatten()

    # randomly return one of the modes
    return vals[np.random.choice(modes_indexes)]

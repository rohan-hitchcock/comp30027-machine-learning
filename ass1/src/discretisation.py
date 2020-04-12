import pandas as pd
import numpy as np
import math

def discretise_equal_width(numeric_series, nbins):
    """ Discretises a numeric data series accoding to equal-width discretisation
        
        Assumes no missing values.

        Args:
            numeric_series: a pd.Series containing numeric data
            nbins: the number of discrete classes for the data
            
        Returns:
            A pd.Series with nbins unique values.
    """

    min_x = numeric_series.min()
    bin_width = (numeric_series.max() - min_x) / nbins

    to_bin_index = lambda x : math.floor((x - min_x) / bin_width)
    return numeric_series.apply(to_bin_index)

def descretise_k_means(numeric_series, k, repeats):
    """ Discretises a numeric data series according to k-means 

        Assumes no missing values.

        Args:
            numeric_series: a pd.Series containing numeric data
            k: the number of discrete categories
            repeates: the number of repeats of k-means
        
        Returns:
            A pd.Series with k unique values of the same index as numeric_series
    """

    #store multiple runs of k-means in a dataframe
    discretised = pd.DataFrame(index=numeric_series.index)

    for _ in range(repeats):

        #TODO: does sorting seeds enough to make sure categories are matched up
        #between runs?
        seeds = np.random.choice(numeric_series, k, replace=False)
        seeds.sort()

        to_discrete = lambda x : np.argmin([abs(x - sd) for sd in seeds])

        discretised = discretised.join(numeric_series.apply(to_discrete))

    #return the most frequent class for each value in numeric series
    return discretised.apply(mode, axis=1)


def mode(xs):
    """ Returns one mode of a sequence xs.

        Args:
            xs: a seqence 

        Returns:
            one of the modes of xs, ignoring NaN values
    """
    vals, counts = np.unique(xs, return_counts=True)

    #find all indices for which the maxium count occurs
    modes_indexes = np.argwhere(counts == np.nanmax(counts)).flatten()

    #randomly return one of the modes
    return vals[np.random.choice(modes_indexes)]

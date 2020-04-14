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

    res = numeric_series.apply(to_bin_index)
    res.name = numeric_series.name
    return res

def descretise_k_means(numeric_series, k, repeats=5):
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

    for run_num in range(repeats):

        #TODO: does sorting seeds enough to make sure categories are matched up
        #between runs?
        centroids = np.random.choice(numeric_series, k, replace=False)
        centroids.sort()

        to_discrete = lambda x : np.argmin([abs(x - sd) for sd in centroids])

        clusters = k * [[]]

        while True:

            new_clusters = k * [[]]

            for x in numeric_series:
                new_clusters[to_discrete(x)].append(x)

            if new_clusters == clusters:
                break

            clusters = new_clusters
            for i, cluster in enumerate(clusters):
                centroids[i] = np.mean(cluster)

        #compute discretization for this iteration of kmeans
        this_disc = numeric_series.apply(to_discrete)           
        this_disc.name = str(run_num)

        discretised = discretised.join(this_disc)

    #return the most frequent class for each value in numeric series
    disc_mode = discretised.apply(mode, axis=1)
    disc_mode.name = numeric_series.name
    return disc_mode


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

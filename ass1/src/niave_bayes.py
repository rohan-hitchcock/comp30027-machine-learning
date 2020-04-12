import numpy as np
import pandas as pd
from collections import namedtuple


""" Convenient way to store the data for a learned Naive Bayes model. 

    Fields:
        discrete:     discrete probability data. A triple-nested dictionary
                      such that discrete[da][c][x] = P(x|c) where x is a 
                      value of a discrete attribute da.
        numeric:      numeric probability data. A dictionary (keyed by attribute 
                      name) of tuples (mu, sigma) where each is an array of the 
                      same length as class_vals storing the mean and standard 
                      deviation of this attribute for each class value.
        class_vals:   an array of possible values of the class to predict
        class_priors: an array the same length of class_vals such that the ith
                      element is the probability of class_vals[i]
"""
NBModel = namedtuple("NBModel", ['discrete', 'numeric', 
                                 'class_vals', 'class_priors'])

def discrete_probabilities(obs, vals):
    """ Estimates the probability of observing each value of a discrete 
        phenomena, based on a series of observations.

        Args:
            obs: a numpy array of observations where each element is contained 
            in vals
            vals: an iterable of possibly observable values

        Returns:
            A numpy array a where a[i] is the probability of observing vals[i]
    """
    return np.array([np.count_nonzero((obs == v)) / len(obs) for v in vals])


def laplace_smoothing(n_attr_obs, n_class_obs, n_attr_vals, alpha):
    return (n_attr_obs + alpha) / (n_class_obs + alpha * n_attr_vals)

# Discrete Naive Bayes ********************************************************
def train_discrete_standard(df, class_attr, class_vals, d_attr, eps=0):
    """ For training Naive Bayes on a discrete attribute, with no / simple 
        smoothing.

        For each class value computes the conditional probability of observing 
        each attribute value given that class value. 

        Args:
            df: a pd.DataFrame
            class_attr: the column in df of the class to predict
            class_vals: the possible values of class_attr
            d_attr: the column in df of the discrete attribute
            eps: (optional) a value to replace any zero probabilities with.

        Returns:
            A dictionary, keyed by class values, of dictonaries storing the 
            conditional probability of observing each attribute value
    """

    attr_vals = df[d_attr].unique()

    params = dict()
    for cv in class_vals:

        params[cv] = dict()
        cv_obs = df[df[class_attr] == cv]
        for av in attr_vals:
            
            num_av_obs = np.count_nonzero((cv_obs[d_attr] == av).to_numpy())

            pval = num_av_obs / cv_obs.shape[0]

            params[cv][av] = pval if pval != 0 else eps

    return params

def train_discrete_laplace(df, class_attr, class_vals, d_attr, alpha):
    """ For training Naive Bayes on a discrete attribute, with laplace smoothing

        For each class value computes the conditional probability of observing 
        each attribute value given that class value. 

        Args:
            df: a pd.DataFrame
            class_attr: the column in df of the class to predict
            class_vals: the possible values of class_attr
            d_attr: the column in df of the discrete attribute
            alpha: the alpha value in laplace smoothing

        Returns:
            A dictionary, keyed by class values, of dictonaries storing the 
            conditional probability of observing each attribute value
    """

    attr_vals = df[d_attr].unique()

    params = dict()
    for cv in class_vals:

        params[cv] = dict()
        cv_obs = df[df[class_attr] == cv]
        for av in attr_vals:
            
            num_av_obs = np.count_nonzero((cv_obs[d_attr] == av).to_numpy())
            params[cv][av] = laplace_smoothing(num_av_obs, cv_obs.shape[0], len(attr_vals), alpha)

    return params

# Guassian Naive Bayes ********************************************************
def train_gaussian(df, class_attr, class_values, num_attr):
    """ For training Naive Bayes on a numeric attribute.
    
        Calculates the mean and standard deviation of a numeric attribute for 
        each class. 

        Args:
            df: A pd.DataFrame 
            class_attr: the column in df of the class we are predicting
            class_values: an iterable of each possible value of class_attr
            num_attr: a numeric attribute in df

        Returns:
            A dictionary (keyed by the class values) of tuples (mean, std)
    """

    means = np.empty(len(class_values))
    stdevs = np.empty(len(class_values))
    for i, cv in enumerate(class_values):

        #A Series which is True wherever cv was observed and false otherwise
        cv_obs = df[class_attr] == cv

        #TODO: write our own functions?
        #by default Series.mean and Series.std will skip missing vals
        means[i] = df[num_attr][cv_obs].mean()
        stdevs[i] = df[num_attr][cv_obs].std()
    
    return (means, stdevs)



# ******************************************************************************
def guassian_pdf(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-(x - mu) ** 2 / 2 * (sigma ** 2)))


def train(df, discrete_attrs, numeric_attrs, class_name, 
          train_discrete=train_discrete_standard, train_numeric=train_gaussian):
    """ Produce a Naive Bayes model for a given dataset.
    
        Args:
            df: a pd.DataFrame of training data
            discrete_attrs: a list of discrete attribute names of df
            numeric_attrs: a list of numeric attribute names of df
            class_name: the name in df of the class to predict
            train_discrete: a callable 
                train_discrete(df, class_name, class_vals, attr) which calculates
                the conditional probabilities of a discrete attribute attr
            train_numeric: a callable 
                train_numeric(df, class_name, class_vals, attr) which calculates
                the means and standard deviations of a numeric attribute attr
        
        Returns:
            A NBModel object
    """
    class_vals = df[class_name].unique()

    discrete = dict()
    for da in discrete_attrs:
        discrete[da] = train_discrete(df, class_name, class_vals, da)

    numeric = dict()
    for na in numeric_attrs:
        numeric[na] = train_numeric(df, class_name, class_vals, na)
    
    class_priors = discrete_probabilities(df[class_name].to_numpy(), class_vals)

    return NBModel(discrete, numeric, class_vals, class_priors)

def predict(df, nbm):
    """ Predict class labels using a Naive Bayes model.

        Args:
            df: a pd.DataFrame storing training instances.
            nbm: a NBModel object trained to predict on instances in df
        Returns:
            A pd.Series of class labels, with index equal to df.index
    """

    #numeric and discrete attributes 
    n_attrs = list(nbm.numeric.keys())
    d_attrs = list(nbm.discrete.keys())

    #means and standard deviations for numeric attributes
    if nbm.numeric:
        means, stdevs = nbm.numeric


    predictions = pd.Series(np.empty(len(df), dtype=nbm.class_vals.dtype), index=df.index)

    for idx, row in df.iterrows():

        class_likelyhoods = np.empty(len(nbm.class_vals))
        for i, cv in enumerate(nbm.class_vals):

            """changed to cv-1 because the class values """
            cl = nbm.class_priors[i]

            if n_attrs:
                for a, x in zip(n_attrs, row[n_attrs]):
                    if pd.notna(x):
                        cl *= guassian_pdf(x, means[a][i], stdevs[a][i])

            if d_attrs:
                for a, x in zip(d_attrs, row[d_attrs]):
                    if pd.notna(x):
                        cl *= nbm.discrete[a][cv][x]

            class_likelyhoods[i] = cl

        predictions[idx] = nbm.class_vals[np.argmax(class_likelyhoods)]
    return predictions

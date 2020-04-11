import numpy as np
import pandas as pd



def conditional_laplace(data, class_col, class_val, attr, attr_val, alpha):
    class_obs = data[data[class_col] == class_val]
    num_attr_vals = len(np.unique(data[attr]))

    return (len(class_obs[class_obs[attr] == attr_val]) + alpha) / (len(class_obs) + alpha * num_attr_vals)


def conditional_no_smoothing(data, class_col, class_val, attr, attr_val):
    class_obs = data[data[class_col] == class_val]


    return len(class_obs[class_obs[attr] == attr_val]) / len(class_obs)



def conditional_eps(data, class_col, class_val, attr, attr_val, eps):
    p = conditional_no_smoothing(data, class_col, class_val, attr, attr_val)
    return p if abs(p) < eps else eps



def discrete_priors(obs, vals):
    """ Estimates the probability of observing each value of a discrete 
        phenomena, based on a series of observations.

        Args:
            obs: a numpy array of observations where each element is contained 
            in vals
            vals: an iterable of possibly observable values

        Returns:
            A dictionary keyed by elements of vals with values the estimates of
            the probability of each val
    """

    return {v: len(obs[obs == v]) / len(obs) for v in vals}


def calculate_conditionals_discrete(data, class_col, conditional=conditional_laplace):
    conditional_probs = dict()
    class_vals = np.unique(data[class_col])
    for a in data.drop(axis=1, labels=[class_col]).columns:

        conditional_probs[a] = dict()

        attr_vals = np.unique(data[a])

        for cv in class_vals:

            conditional_probs[a][cv] = dict()
            for av in attr_vals:
                conditional_probs[a][cv][av] = conditional(data, class_col, cv, a, av, 1)

    return conditional_probs


def laplace_smoothing(n_attr_obs, n_class_obs, n_attr_vals, alpha):
    return (n_attr_obs + alpha) / (n_class_obs + alpha * n_attr_vals)

# Discrete Naive Bayes ********************************************************
def train_discrete(df, class_attr, class_vals, d_attr, eps=0):
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
    #NOTE: passing class_values since we will need to call this method lots and 
    #it doesn't change

    params = dict()
    for cv in class_values:
        cv_obs = df[class_attr] == cv
        params[cv] = (df[num_attr][cv_obs].mean(), df[num_attr][cv_obs].std())
    
    #TODO: consider changing this to return an array / series ?
    return params



# ******************************************************************************
def guassian_density(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp((-(x - mu) ** 2 / 2 * (sigma ** 2)))



def predict(df, d_attrs, d_pr, n_attrs, n_pr, class_vals, class_piors):

    ddf = df[d_attrs]
    ndf = df[n_attrs]

    predictions = pd.Series(np.empty(len(df), dtype=class_vals.dtype), index=df.index)

    for i, row in df.rows():

        class_probs = np.empty(len(class_vals))
        for i, cv in class_vals:
            

            mu, sigma = n_pr
            

            n_part = np.prod([
                guassian_density(x, mu[a][cv], sigma[a][cv]) for a, x in zip(n_attrs, row[n_attrs])
            ])

            d_part = np.prod([
                d_pr[a][cv][x] for a, x in zip(d_attrs, row[d_attrs]) 
            ])

            class_probs[i] = class_piors[cv] * n_part * d_part

        predictions[i] = class_vals[np.argmax(class_probs)]
    return predictions

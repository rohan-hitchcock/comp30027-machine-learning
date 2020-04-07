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


def train(data, cnfg, conditional=conditional_laplace):
    class_col = cnfg['class_col']
    model = calculate_conditionals_discrete(data, class_col, conditional)
    priors = discrete_priors(data[class_col], np.unique(data[class_col]))
    return model, priors

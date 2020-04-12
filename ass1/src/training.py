import src.niave_bayes as nb
import numpy as np
import pandas as pd

NUM_PARTITIONS = 10


def train(data, cnfg, conditional=nb.conditional_laplace):
    """ given a data set, train returns a list of 3 tuples including the niave bayers model,
        discrete priors, and the test set according to the cross validation partitions"""
    class_col = cnfg['class_col']
    models = list()
    model = nb.calculate_conditionals_discrete(data, class_col, conditional)
    priors = nb.discrete_priors(data[class_col], np.unique(data[class_col]))
    return model, priors




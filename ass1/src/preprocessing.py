import numpy as np
import pandas as pd
import src.config as config

"""Honestly not sure how this is going to work. Seeing as we are submitting a jupyter notebook,
    there isnt even really a "Main" file. We should still define a config dictionary and use
    the filename to determine which config to use. Anyway, worry about that later."""

""" From piazza I found that we are allowed to use sklearn for K-means to discretize numeric attributes"""


def preprocess(filepath):
    if filepath.endswith('lymphography.data'):
        return preprocess_lymphography(filepath, config.config["lymphography"])
    if filepath.endswith('university.data'):
        return preprocess_university(filepath, config.config["university"])


def preprocess_lymphography(filepath, cnfg):
    df = pd.read_csv(filepath, header=cnfg["header"], names=range(cnfg['attributes']))
    # Preprocess and return
    return df, cnfg


def preprocess_university(filepath, cnfg):
    df = pd.read_csv(filepath, header=cnfg["header"], names=range(cnfg['attributes']))
    #TODO imputate missing values (0) with nans
    return df, cnfg

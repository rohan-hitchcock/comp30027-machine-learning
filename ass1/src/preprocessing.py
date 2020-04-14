import numpy as np
import pandas as pd
import src.config as config

import os

"""Honestly not sure how this is going to work. Seeing as we are submitting a jupyter notebook,
    there isnt even really a "Main" file. We should still define a config dictionary and use
    the filename to determine which config to use. Anyway, worry about that later."""

""" From piazza I found that we are allowed to use sklearn for K-means to discretize numeric attributes"""


def preprocess(filepath, print_summary=False):

    #extract the name of the dataset from the filepath
    pth, fname = os.path.split(filepath)    #pylint:disable=unused-variable
    fname, ext = os.path.splitext(fname)    #pylint:disable=unused-variable

    if fname not in config.config:
        raise NotImplementedError(f"Learning on the \'{fname}\' dataset is not supported.")
    
    data_config = config.config[fname]

    df = pd.read_csv(
        filepath, 
        header=data_config['header'], 
        names=range(data_config['attributes']),
        na_values=data_config['missing_values']
    )

    if print_summary:
        summarise(df, data_config)

    return df, data_config
    


def preprocess_lymphography(filepath, cnfg):
    df = pd.read_csv(filepath, header=cnfg["header"], names=range(cnfg['attributes']))
    # Preprocess and return
    return df, cnfg


def preprocess_university(filepath, cnfg):
    df = pd.read_csv(filepath, header=cnfg["header"], names=range(cnfg['attributes']))
    #TODO imputate missing values (0) with nans
    
    
    #TODO:remove this in final version
    class_groups = df.groupby(cnfg['class_col']).groups
    class_vals = list(class_groups.keys())
    class_vals.sort()

    hist = {cv : len(obs) for cv, obs in class_groups.items()}

    hist_marker = "*"
    for cv in class_vals:
        print(f"{cv} | {hist[cv] * hist_marker}\t({hist[cv]})")



    return df, cnfg


def preprocess_adult(filepath, cnfg):
    df = pd.read_csv(
        filepath, 
        header=cnfg["header"], 
        names=range(cnfg['attributes']),
        na_values=cnfg['missing_values'])
    return df, cnfg

def summarise(df, cnfg):
    print("Data summary: -------------------------------------------------------")
    for col in df.columns:
        
        if col == cnfg['class_col']:
            label = "class"
            

        elif col in cnfg['discrete']:
            label = "discrete"

        elif col in cnfg['numeric']:
            label = 'numeric'
        
        #column data is not relavent (eg is an ID)
        else:
            continue

        print (f"{col}: ({label}) dtype={df.dtypes[col]}")
        if label == "discrete" or label == "class":
            vals = df[col].unique()
            print (f"values ({len(vals)}): {vals}")

        num_missing = np.count_nonzero(df[col].isna())
        print(f"missing: {num_missing} / {len(df[col])}\n")
    print("---------------------------------------------------------------------")

    print(df)

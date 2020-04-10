import numpy as np
import pandas as pd
import src.config as config

"""Honestly not sure how this is going to work. Seeing as we are submitting a jupyter notebook,
    there isnt even really a "Main" file. We should still define a config dictionary and use
    the filename to determine which config to use. Anyway, worry about that later."""

def preprocess(filepath):
    if filepath.endswith('lymphography.data'):
        return preprocess_lymphography(filepath, config.lymphography)


def preprocess_lymphography(filepath, cnfg):
    df = pd.read_csv(filepath, header=None, names=range(cnfg['attributes'] + 1))
    # Preprocess and return
    return df, cnfg

import numpy as np
import pandas as pd
import ass1.src.config as config


def preprocess(filepath):
    if filepath.endswith('lymphography.data'):
        return preprocess_lymphography(config.lymphography)


def preprocess_lymphography(cnfg):
    df = pd.read_csv(cnfg['filepath'], header=None, names=range(cnfg['attributes'] + 1))
    # Preprocess and return
    return df, cnfg

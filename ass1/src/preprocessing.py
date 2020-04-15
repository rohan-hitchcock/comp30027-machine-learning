import numpy as np
import pandas as pd
import src.config as config

import os


def preprocess(filepath, print_summary=False):
    # extract the name of the dataset from the filepath
    pth, fname = os.path.split(filepath)  # pylint:disable=unused-variable
    fname, ext = os.path.splitext(fname)  # pylint:disable=unused-variable

    if fname not in config.config:
        raise NotImplementedError(f"Learning on the \'{fname}\' dataset is not supported.")

    data_config = config.config[fname]

    df = pd.read_csv(
        filepath,
        header=data_config['header'],
        names=range(data_config['attributes']),
        na_values=data_config['missing_values']
    )

    df = df.sample(frac=1, random_state=2).reset_index(drop=True)

    if print_summary:
        summarise(df, data_config)

    return df, data_config


def summarise(df, cnfg):
    print("Data summary: -------------------------------------------------------")
    for col in df.columns:

        if col == cnfg['class_col']:
            label = "class"

        elif col in cnfg['discrete']:
            label = "discrete"

        elif col in cnfg['numeric']:
            label = 'numeric'

        # column data is not relavent (eg is an ID)
        else:
            continue

        print(f"{col}: ({label}) dtype={df.dtypes[col]}")
        if label == "discrete" or label == "class":
            vals = df[col].unique()
            print(f"values ({len(vals)}): {vals}")

        num_missing = np.count_nonzero(df[col].isna())
        print(f"missing: {num_missing} / {len(df[col])}\n")
    print("---------------------------------------------------------------------")

    print(df)

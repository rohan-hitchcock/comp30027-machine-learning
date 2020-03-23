import pandas as pd
import numpy as np
from ass1.src.config import lymphography
import ass1.src.niave_bayes as nb


# def format_priors(priors):

def main():
    df = pd.read_csv('../datasets/lymphography.data', header=None, names=range(lymphography['attributes'] + 1))
    print(nb.calculate_conditionals_discrete(df, lymphography['class_col'], lymphography['class_vals']))


if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from ass1.src.config import lymphography
import ass1.src.niave_bayes as nb
import ass1.src.predict as predict


# def format_priors(priors):

def main():
    df = pd.read_csv('../datasets/lymphography.data', header=None, names=range(lymphography['attributes'] + 1))
    class_col = lymphography['class_col']
    model = nb.calculate_conditionals_discrete(df, class_col)
    class_vals = np.unique(df[class_col])
    priors = nb.discrete_priors(df[class_col], class_vals)
    print(predict.predict(df.drop(axis=1, labels=[class_col]).to_numpy(), model, class_vals, priors))
    print(np.array(df[class_col]) == predict.predict(df.drop(axis=1, labels=[class_col]).to_numpy(), model, class_vals,
                                                 priors))


if __name__ == '__main__':
    main()

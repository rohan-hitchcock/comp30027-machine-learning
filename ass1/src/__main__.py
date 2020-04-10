import sys
import pandas as pd
import numpy as np

import src.niave_bayes as nb
import src.predict as predict
import src.preprocessing as pprcs
import src.evaluate as ev
import src.baseline as bl

# TODO: change main to take in filepointer as argument

def main():
    """Obviously this will change since were submitting a jupyter notebook, but for now run
        this with:
        python3 -m src {filepath}

        Must be from within the ass1 folder
        """

    data, cnfg = pprcs.preprocess(sys.argv[1])
    model, priors = nb.train(data, cnfg)
    ybar = predict.predict(data, model, cnfg, priors)
    class_col = data[cnfg['class_col']]
    print("---------- Zero R ----------")
    ev.evaluate(class_col, bl.zero_r(class_col))
    print("---------- Model Evaluation ----------")
    ev.evaluate(class_col, ybar)

if __name__ == '__main__':
    main()

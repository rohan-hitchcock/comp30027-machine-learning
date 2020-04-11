import sys
import pandas as pd
import numpy as np
import src.predict as predict
import src.preprocessing as pprcs
import src.evaluate as ev
import src.baseline as bl
import src.training as train

# TODO: change main to take in filepointer as argument

def main():
    """Obviously this will change since were submitting a jupyter notebook, but for now run
        this with:
        python3 -m src {filepath}

        Must be from within the ass1 folder
        """

    data, cnfg = pprcs.preprocess(sys.argv[1])

    """ this now is a list of 3 tuples. The model, priors and test set"""
    models = train.train(data, cnfg)

    """This is now a list of 2 tuples. The ground truth and predicted labels for
        each test partition"""
    ybars = list()
    for model, priors, test in models:
        ybars.append(predict.predict(test, model, cnfg, priors))


    """ Calculating the fscores for each, but there are more metrics within evaluate file"""
    fscores = list()
    for truth, ybar in ybars:
        fscores.append(ev.evaluate(truth, ybar))

    """prints the mean"""
    print(np.mean(fscores))

    """Old stuff before I used cross validation"""
    # class_col = data[cnfg['class_col']]
    # ybar = predict.predict(data, model, cnfg, priors)
    # print("---------- Zero R ----------")
    # ev.evaluate(class_col, bl.zero_r(class_col))
    # print("---------- Model Evaluation ----------")
    # ev.evaluate(class_col, ybar)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy as np
from ass1.src.config import lymphography
import ass1.src.niave_bayes as nb
import ass1.src.predict as predict
import ass1.src.preprocessing as pprcs
import ass1.src.evaluate as ev
import ass1.src.baseline as bl

# TODO: change main to take in filepointer as argument

def main():
    lymph, lymph_cnfg = pprcs.preprocess('../datasets/lymphography.data')
    model, priors = nb.train(lymph, lymph_cnfg)
    ybar = predict.predict(lymph, model, lymph_cnfg, priors)
    class_col = lymph[lymph_cnfg['class_col']]
    print("---------- Zero R ----------")
    ev.evaluate(class_col, bl.zero_r(class_col))
    print("---------- Model Evaluation ----------")
    ev.evaluate(class_col, ybar)

if __name__ == '__main__':
    main()

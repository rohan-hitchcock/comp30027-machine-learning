import pandas as pd
import numpy as np
from ass1.src.config import lymphography
import ass1.src.niave_bayes as nb
import ass1.src.predict as predict
import ass1.src.preprocessing as pprcs


# def format_priors(priors):

def main():
    lymph, lymph_cnfg = pprcs.preprocess('../datasets/lymphography.data')
    class_col = lymph_cnfg['class_col']
    model, priors = nb.train(lymph, lymph_cnfg)
    print(predict.predict(lymph, model, lymph_cnfg, priors))
    print(np.array(lymph[class_col]) == predict.predict(lymph, model, lymph_cnfg, priors))


if __name__ == '__main__':
    main()

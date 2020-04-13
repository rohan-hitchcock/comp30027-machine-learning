import sys
import src.preprocessing as pprcs
import src.evaluate as ev
import src.baseline as bl
import src.niave_bayes as nb
import src.cross_validation as cv

NUM_PARTITIONS = 10


def main():
    """Obviously this will change since were submitting a jupyter notebook, but for now run
        this with:
        python3 -m src {filepath}

        Must be from within the ass1 folder
        """

    
    print(sys.argv[1])
    df, cnfg = pprcs.preprocess(sys.argv[1])


    print("---------- Cross-Val Evaluation ----------")
    cv_results = cv.cross_validation(df, cnfg, NUM_PARTITIONS)
    ev.print_eval(cv_results[0], cv_results[1], cv_results[2], cv_results[3])

    model = nb.train(df, cnfg["discrete"], cnfg["numeric"], cnfg["class_col"])

    predictions = nb.predict(df, model)
    truth = df[cnfg["class_col"]]
    print("---------- Model Evaluation ----------")
    ev.evaluate(truth, predictions)
    print("---------- Zero R ----------")
    zero_r = bl.classify_zero_r(truth, cnfg["instances"])
    ev.evaluate(truth, zero_r)

if __name__ == '__main__':
    main()

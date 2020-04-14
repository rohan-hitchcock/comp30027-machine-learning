import sys
import src.preprocessing as pprcs
import src.evaluate as ev
import src.baseline as bl
import src.niave_bayes as nb
import src.cross_validation as cv
import src.discretisation as disc
import numpy as np

NUM_PARTITIONS = 10


def main():
    """Obviously this will change since were submitting a jupyter notebook, but for now run
        this with:
        python3 -m src {filepath}

        Must be from within the ass1 folder
        """

    
    print(sys.argv[1])
    df, cnfg = pprcs.preprocess(sys.argv[1], print_summary=True)

    print("---------- Cross-Val Evaluation ----------")
    cv_results = cv.cross_validation(df, cnfg, NUM_PARTITIONS)
    ev.print_eval(cv_results[0], cv_results[1], cv_results[2], cv_results[3])

    print("---------- Model Evaluation ----------")
    model = nb.train(df, cnfg["discrete"], cnfg["numeric"], cnfg["class_col"], train_discrete=nb.train_discrete_laplace)
    predictions = nb.predict(df, model)
    truth = df[cnfg["class_col"]]
    ev.evaluate(truth, predictions)

    if cnfg["numeric"]:
        
        num_discrete = 5

        print(f"------ Discretisation ({num_discrete})----------")

        print("K-means:")
        kmeans_df = df[cnfg["discrete"]].join(df[cnfg["class_col"]])

        for num_attr in cnfg["numeric"]:

            discretized = disc.descretise_k_means(df[num_attr], num_discrete, repeats=1)
            kmeans_df = kmeans_df.join(discretized)

            print(len(kmeans_df[num_attr].unique()))

        kmeans_cnfg = cnfg.copy()

        kmeans_cnfg["discrete"] = cnfg["discrete"] + cnfg["numeric"]
        kmeans_cnfg["numeric"] = []

        print(kmeans_cnfg)

        cv_results = cv.cross_validation(kmeans_df, kmeans_cnfg, NUM_PARTITIONS)
        ev.print_eval(cv_results[0], cv_results[1], cv_results[2], cv_results[3])

        print("\nEqual Width:")


        eqw_df = df[cnfg["discrete"]].join(df[cnfg["class_col"]])

        for num_attr in cnfg["numeric"]:

            discretized = disc.discretise_equal_width(df[num_attr], num_discrete)
            eqw_df = eqw_df.join(discretized)

            print(len(eqw_df[num_attr].unique()))
            

        eqw_cnfg = cnfg.copy()
        eqw_cnfg["discrete"] = cnfg["discrete"] + cnfg["numeric"]
        eqw_cnfg["numeric"] = []

        cv_results = cv.cross_validation(eqw_df, eqw_cnfg, NUM_PARTITIONS)
        ev.print_eval(cv_results[0], cv_results[1], cv_results[2], cv_results[3])
            

    print("---------- Baseline ----------")
    zero_r = bl.classify_zero_r(truth, cnfg["instances"])
    random = bl.classify_random(truth, cnfg["instances"])
    uniform = bl.classify_uniform(truth, cnfg['instances'])
    one_r = bl.classify_one_r(df, cnfg['class_col'], df)
    print("Zero-R:")
    ev.evaluate(truth, zero_r)
    print("Random:")
    ev.evaluate(truth, random)
    print("Uniform:")
    ev.evaluate(truth, uniform)
    print("One-R:")
    ev.evaluate(truth, one_r)
    

if __name__ == '__main__':
    main()

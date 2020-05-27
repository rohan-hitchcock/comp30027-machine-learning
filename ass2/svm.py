from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
import itertools

from polarized_svm import PolarizedSVM
from generate_docvecs import get_dot2vec_split
from generate_docvecs import get_doc2vec_crossval

Evaluation = namedtuple("Evaluation", ["accuracy", "fscore", "precision", "recall"])

def evaluate_model(X_file, y, model, num_xval_splits=3):
    """ Evaluate a model on a dataset (X, y) using cross validaiton.

        Args:
            X: instances (pd.DataFrame)
            y: class labels (pd.DataFrame or pd.Series)
            model: an instance of a class with methods .fit and .predict
            num_xval_splits: the number of cross validaiton splits to use

        Returns: A tuple of fscore, accuracy, precision and recall averaged over
        the cross validaiton splits
    """
    X = pd.read_csv(X_file, index_col=False, delimiter=',', header=None
    )
    cross_val = KFold(n_splits=num_xval_splits)

    fscore_tst = np.empty(num_xval_splits)
    accuracy_tst = np.empty(num_xval_splits)
    precision_tst = np.empty(num_xval_splits)
    recall_tst =  np.empty(num_xval_splits)

    fscore_trn = np.empty(num_xval_splits)
    accuracy_trn = np.empty(num_xval_splits)
    precision_trn = np.empty(num_xval_splits)
    recall_trn = np.empty(num_xval_splits)

    for i, split in enumerate(cross_val.split(X)):
        train_i, test_i = split

        data_train, data_test = X.iloc[train_i], X.iloc[test_i]
        class_label_train, class_label_test = y.iloc[train_i], y.iloc[test_i]
        
        model.fit(data_train, class_label_train)

        #prediction
        predict_label = model.predict(data_test)

        #evaluation
        fscore_tst[i] = metrics.f1_score(class_label_test, predict_label, average='weighted')
        accuracy_tst[i] = metrics.accuracy_score(class_label_test, predict_label)
        precision_tst[i] = metrics.precision_score(class_label_test, predict_label, average='weighted')
        recall_tst[i] = metrics.recall_score(class_label_test, predict_label, average='weighted')

        predict_label = model.predict(data_train)

        #evaluation
        fscore_trn[i] = metrics.f1_score(class_label_train, predict_label, average='weighted')
        accuracy_trn[i] = metrics.accuracy_score(class_label_train, predict_label)
        precision_trn[i] = metrics.precision_score(class_label_train, predict_label, average='weighted')
        recall_trn[i] = metrics.recall_score(class_label_train, predict_label, average='weighted')

    del X

    return (
        Evaluation(np.mean(accuracy_tst), np.mean(fscore_tst), np.mean(precision_tst), np.mean(recall_tst)), 
        Evaluation(np.mean(accuracy_trn), np.mean(fscore_trn), np.mean(precision_trn), np.mean(recall_trn))
    ) 

def compute_class_proportions():
    data_meta = pd.read_csv(r"./datasets/review_meta_train.csv")
    class_label = data_meta['rating']


    print("original dataset")
    num_instances = len(class_label)
    one_count = np.count_nonzero(class_label == 1)
    three_count = np.count_nonzero(class_label == 3)
    five_count = np.count_nonzero(class_label == 5)
    print(f"Count of 1: {one_count} ({round(100 * one_count / num_instances, 3)} pc)")
    print(f"Count of 3: {three_count} ({round(100 * three_count / num_instances, 3)} pc)")
    print(f"Count of 5: {five_count} ({round(100 * five_count / num_instances, 3)} pc)")
    print(f"Total: {num_instances}")

    print("-------------------")
    for dim in range(25, 301, 25):

        print(f"for dimension {dim}")
        Xtrain, Xtest, ytrain, ytest = get_dot2vec_split(dim)

        print("train:")
        num_instances = len(ytrain)
        one_count = np.count_nonzero(ytrain == 1)
        three_count = np.count_nonzero(ytrain == 3)
        five_count = np.count_nonzero(ytrain == 5)

        print(f"Count of 1: {one_count} ({round(100 * one_count / num_instances, 3)} pc)")
        print(f"Count of 3: {three_count} ({round(100 * three_count / num_instances, 3)} pc)")
        print(f"Count of 5: {five_count} ({round(100 * five_count / num_instances, 3)} pc)")
        print(f"Total: {num_instances}")

        print("test:")
        num_instances = len(ytest)
        one_count = np.count_nonzero(ytest == 1)
        three_count = np.count_nonzero(ytest == 3)
        five_count = np.count_nonzero(ytest == 5)

        print(f"Count of 1: {one_count} ({round(100 * one_count / num_instances, 3)} pc)")
        print(f"Count of 3: {three_count} ({round(100 * three_count / num_instances, 3)} pc)")
        print(f"Count of 5: {five_count} ({round(100 * five_count / num_instances, 3)} pc)")
        print(f"Total: {num_instances}")

        print("-------------------")

def learning_curve(model, fname_test, fname_train):


    with open(fname_test, "a") as fp:
        fp.write("dim, fscore, accuracy, precision, recall\n" )

    with open(fname_train, "a") as fp:
        fp.write("dim, fscore, accuracy, precision, recall\n" )


    for dim in range(25, 301, 25):
        
        print(f"dim = {dim}")

        Xtrain, Xtest, ytrain, ytest = get_dot2vec_split(dim)

        model.fit(Xtrain, ytrain)

        predictions = model.predict(Xtest)

        fscore = metrics.f1_score(ytest, predictions, average='weighted')
        accuracy = metrics.accuracy_score(ytest, predictions)
        precision = metrics.precision_score(ytest, predictions, average='weighted')
        recall =  metrics.recall_score(ytest, predictions, average='weighted')

        with open(fname_test, 'a') as fp:
            fp.write(f"{dim}, {fscore}, {accuracy}, {precision}, {recall}\n")

        predictions = model.predict(Xtrain)

        fscore = metrics.f1_score(ytrain, predictions, average='weighted')
        accuracy = metrics.accuracy_score(ytrain, predictions)
        precision = metrics.precision_score(ytrain, predictions, average='weighted')
        recall = metrics.recall_score(ytrain, predictions, average='weighted')

        with open(fname_train, 'a') as fp:
            fp.write(f"{dim}, {fscore}, {accuracy}, {precision}, {recall}\n")

        del Xtrain
        del Xtest
        del ytrain
        del ytest

def learning_curve_rbf(c, gamma, fname_test, fname_train):


    with open(fname_test, "a") as fp:
        fp.write("dim, fscore, accuracy, precision, recall\n" )

    with open(fname_train, "a") as fp:
        fp.write("dim, fscore, accuracy, precision, recall\n" )


    for dim in range(25, 301, 25):
        
        print(f"dim = {dim}")

        Xtrain, Xtest, ytrain, ytest = get_dot2vec_split(dim)

        true_gamma = gamma / (np.array(Xtrain).var() * len(Xtrain.columns))
        model = svm.SVC(kernel='rbf', C=c, gamma=true_gamma)

        model.fit(Xtrain, ytrain)

        predictions = model.predict(Xtest)

        fscore = metrics.f1_score(ytest, predictions, average='weighted')
        accuracy = metrics.accuracy_score(ytest, predictions)
        precision = metrics.precision_score(ytest, predictions, average='weighted')
        recall =  metrics.recall_score(ytest, predictions, average='weighted')

        with open(fname_test, 'a') as fp:
            fp.write(f"{dim}, {fscore}, {accuracy}, {precision}, {recall}\n")

        predictions = model.predict(Xtrain)

        fscore = metrics.f1_score(ytrain, predictions, average='weighted')
        accuracy = metrics.accuracy_score(ytrain, predictions)
        precision = metrics.precision_score(ytrain, predictions, average='weighted')
        recall = metrics.recall_score(ytrain, predictions, average='weighted')

        with open(fname_train, 'a') as fp:
            fp.write(f"{dim}, {fscore}, {accuracy}, {precision}, {recall}\n")

        del Xtrain
        del Xtest
        del ytrain
        del ytest


def gridsearch_c(param_space, dim, kernel, xval_size=5):

    for i, c in enumerate(param_space):
        
        print(f"({i}/{len(param_space)}) values searched (C={c}).")

        model = svm.SVC(kernel=kernel, C=c)

        fscore = 0
        accuracy = 0
        precision = 0
        recall = 0
        xval_num = 0
        for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, xval_size):
            
            xval_num += 1
            print(f"\t{xval_num}/{xval_size}")

            print("fitting model....")
            model.fit(Xtrain, ytrain)

            print("predicting....")
            predictions = model.predict(Xtest)

            print("computing metric....")
            fscore += metrics.f1_score(ytest, predictions, average='weighted')
            accuracy += metrics.accuracy_score(ytest, predictions)
            precision += metrics.precision_score(ytest, predictions, average='weighted')
            recall += metrics.recall_score(ytest, predictions, average='weighted')

            del Xtrain
            del Xtest
            del ytrain
            del ytest

        with open(f"./results/svm/gridsearch_{kernel}_{dim}.csv", "a") as fp:
            fp.write(f"{c}, {fscore / xval_size}, {accuracy / xval_size}, {precision / xval_size}, {recall / xval_size}\n")


def gridsearch_polar(param_space, dim, xval_size=5):

    for i, param in enumerate(param_space):
        c, pthresh = param
        print(f"({i}/{len(param_space)}) values searched (C={c}, pthresh={pthresh}).")

        model = PolarizedSVM(pthresh, 3, C=c)

        fscore = 0
        accuracy = 0
        precision = 0
        recall = 0
        xval_num = 0
        for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, xval_size):
            
            xval_num += 1
            print(f"\t{xval_num}/{xval_size}")

            print("fitting model....")
            model.fit(Xtrain, ytrain)

            print("predicting....")
            predictions = model.predict(Xtest)

            print("computing metric....")
            fscore += metrics.f1_score(ytest, predictions, average='weighted')
            accuracy += metrics.accuracy_score(ytest, predictions)
            precision += metrics.precision_score(ytest, predictions, average='weighted')
            recall += metrics.recall_score(ytest, predictions, average='weighted')

            del Xtrain
            del Xtest
            del ytrain
            del ytest

        with open(f"./results/svm/gridsearch_polar_{dim}.csv", "a") as fp:
            fp.write(f"{c}, {pthresh}, {fscore / xval_size}, {accuracy / xval_size}, {precision / xval_size}, {recall / xval_size}\n")

def gridsearch_c_gamma(param_space, dim, kernel, xval_size=5):

    for i, param in enumerate(param_space):
        c, gamma = param
        print(f"({i}/{len(param_space)}) values searched (C={c}). gamma={gamma}")

        

        fscore = 0
        accuracy = 0
        precision = 0
        recall = 0
        xval_num = 0
        for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, xval_size):
            
            true_gamma = gamma / (np.array(Xtrain).var() * len(Xtrain.columns))

            model = svm.SVC(kernel=kernel, C=c, gamma=true_gamma)


            xval_num += 1
            print(f"\t{xval_num}/{xval_size}")

            print("fitting model....")
            model.fit(Xtrain, ytrain)

            print("predicting....")
            predictions = model.predict(Xtest)

            print("computing metric....")
            fscore += metrics.f1_score(ytest, predictions, average='weighted')
            accuracy += metrics.accuracy_score(ytest, predictions)
            precision += metrics.precision_score(ytest, predictions, average='weighted')
            recall += metrics.recall_score(ytest, predictions, average='weighted')

            del Xtrain
            del Xtest
            del ytrain
            del ytest

        with open(f"./results/svm/gridsearch_{kernel}_{dim}.csv", "a") as fp:
            fp.write(f"{c}, {gamma}, {fscore / xval_size}, {accuracy / xval_size}, {precision / xval_size}, {recall / xval_size}\n")


print("learning curve for rbf...")
C = 1.25
gamma = 0.6
learning_curve_rbf(C, gamma, "./results/svm/learning_curve_rbf_final_test.csv", "./results/svm/learning_curve_rbf_final_train.csv")


print("confusion matrix for rbf...")
cm = np.zeros((3, 3))
xval_size = 5
dim = 125
for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, xval_size):

    true_gamma = gamma / (np.array(Xtrain).var() * len(Xtrain.columns))
    model = svm.SVC(kernel='rbf', C=C, gamma=true_gamma)
    model.fit(Xtrain, ytrain)

    predictions = model.predict(Xtest)

    cm += metrics.confusion_matrix(ytest, predictions)

cm = cm / xval_size
np.savetxt("./results/svm/cm_rbf_final.csv", cm)


print("learning curve for binary...")
model = PolarizedSVM(threshold=0.9, middle_class=3, C=0.0025)
learning_curve(model, "./results/svm/learning_curve_binary_final_test.csv", "./results/svm/learning_curve_binary_final_train.csv")

print("confusion matrix for binary...")
cm = np.zeros((3, 3))
xval_size = 5
dim = 125
for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, xval_size):
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)
    cm += metrics.confusion_matrix(ytest, predictions)

cm = cm / xval_size
np.savetxt("./results/svm/cm_binary_final.csv", cm)

print("Confusion matrix for linear....")
model = svm.SVC(kernel='linear', C=0.009)
cm = np.zeros((3, 3))
xval_size = 5
dim = 150
for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, xval_size):
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)
    cm += metrics.confusion_matrix(ytest, predictions)

cm = cm / xval_size
np.savetxt("./results/svm/cm_linear_final.csv", cm)



"""
param_space = list(itertools.product([1, 1.25], [0.2, 0.1]))
gridsearch_c_gamma(param_space, dim, kernel)
"""


"""
param_space = list(itertools.product([0.0005, 0.001, 0.0015, 0.002], [0.95]))

param_space = list(itertools.product([0.0005, 0.001, 0.0015, 0.002], [0.95]))
gridsearch_polar(pa am_space, dim)


param_space = list(itertools.product([0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005], [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]))
gridsearch_polar(param_space, dim)
"""


"""
print("Original Hold Out:")

Xtrain, Xtest, ytrain, ytest = get_dot2vec_split(150)
print(Xtrain.head(10))
print(ytrain.head(10))

print(f"Xtrain count: {len(Xtrain)}")
print("\n-------------------------------------------------")

counter = 1
for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(150):
    print(f"Cross val {counter}:")
    print(Xtrain.head(10))
    print(ytrain.head(10))

    print(f"Xtrain count: {len(Xtrain)}")


    num_instances = len(ytest)
    one_count = np.count_nonzero(ytest == 1)
    three_count = np.count_nonzero(ytest == 3)
    five_count = np.count_nonzero(ytest == 5)

    print(f"Count of 1: {one_count} ({round(100 * one_count / num_instances, 3)} pc)")
    print(f"Count of 3: {three_count} ({round(100 * three_count / num_instances, 3)} pc)")
    print(f"Count of 5: {five_count} ({round(100 * five_count / num_instances, 3)} pc)")
    print(f"Total: {num_instances}")

    print("\n-------------------------------------------------")

    counter += 1
"""

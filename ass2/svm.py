from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt

from generate_docvecs import get_dot2vec_split

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


def feature_dim_learning_curve(feature_sets, class_label, model, x_vals, y_metric_name, y_label, title):

    y_vals_test = []
    y_vals_train = []
    for i, X in enumerate(feature_sets):

        print(f"run {i}")

        test_eval, train_eval = evaluate_model(X, class_label, model)

        y_vals_test.append(getattr(test_eval, y_metric_name))
        y_vals_train.append(getattr(train_eval, y_metric_name))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(x_vals, y_vals_test, 'bo', linestyle='-', label='Test')
    ax.plot(x_vals, y_vals_test, 'go', linestyle='-', label='Train')
    
    ax.set_xaxis('Feature space dimension')
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.legend(loc='upper left', title_fontsize=12)

    plt.show()




"""
data50 = pd.read_csv(
    r"./datasets/review_text_features_doc2vec50/review_text_train_doc2vec50.csv", 
    index_col=False, delimiter=',', header=None
)

data100 = pd.read_csv(
    r"./datasets/review_text_features_doc2vec100/review_text_train_doc2vec100.csv", 
    index_col=False, delimiter=',', header=None
)

data200 = pd.read_csv(
    r"./datasets/review_text_features_doc2vec200/review_text_train_doc2vec200.csv", 
    index_col=False, delimiter=',', header=None
)
"""
"""
data50 =  r"./datasets/review_text_features_doc2vec50/review_text_train_doc2vec50.csv"
data100 = r"./datasets/review_text_features_doc2vec100/review_text_train_doc2vec100.csv"
data200 = r"./datasets/review_text_features_doc2vec200/review_text_train_doc2vec200.csv"

"""

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



"""
kernal = 'linear'
model = svm.SVC(C=1, kernel=kernal)

feature_sets = [data50, data100, data200]
x_vals = [50, 100, 200]

y_metric_name = 'fscore'
y_label = "F-Score"
title = "Linear SVM"

feature_dim_learning_curve(feature_sets, class_label, model, x_vals, y_metric_name, y_label, title)
"""

"""
data = data50
data_meta = pd.read_csv(r"./datasets/review_meta_train.csv")
class_label = data_meta['rating']

for kernal in ['linear', 'poly', 'rbf', 'sigmoid']:
    
    model = svm.SVC(C=1, kernel=kernal)

    eval_test, eval_train = evaluate_model(data, class_label, model)
    print(kernal)
    print(f"Accuracy: {eval_test.accuracy}\n"
          f"Precision: {eval_test.precision}\n"
          f"Recall: {eval_test.recall}\n"
          f"F1Score: {eval_test.fscore}")

    print()
"""

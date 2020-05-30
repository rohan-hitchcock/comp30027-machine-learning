from sklearn.ensemble import StackingClassifier
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from generate_docvecs import get_doc2vec_crossval
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlr import autolabel

# SVM
# linear: C=0.009, dim=150
# RBF: C=1.25, gamma=0.6, dim=125
# polar: C=0.0025, thresh=0.9, dim=125
# Logistic Regression: C=0.015, dim=150
from sklearn.metrics import f1_score, accuracy_score

from plotting import plot_confusion_matrix

RANDOM_STATE = 7

# ---------- Comparing Accuracy and Fscore for all combinations of stacked classifiers
def stacking(n_splits=5):
    eval_dict = {
        "clf": [],
        "fscore": [],
        "accuracy": []
    }

    stack_linear = StackingClassifier(
        estimators=[('LogReg', LogisticRegression(max_iter=200, C=0.015)),
                    ('LinearSVM', svm.SVC(kernel='linear', C=0.009))],
        final_estimator=LogisticRegression(max_iter=200))
    stack_svm = StackingClassifier(
        estimators=[('LinearSVM', svm.SVC(kernel='linear', C=0.009)),
                    ('RBFSVM', svm.SVC(kernel='rbf', C=1.25))],
        final_estimator=LogisticRegression(max_iter=200))
    stack_rbf = StackingClassifier(
        estimators=[('LogReg', LogisticRegression(max_iter=200, C=0.015)),
                    ('RBFSVM', svm.SVC(kernel='rbf', C=1.25))],
        final_estimator=LogisticRegression(max_iter=200))
    stack_all = StackingClassifier(
        estimators=[('LinearSVM', svm.SVC(kernel='linear', C=0.009)),
                    ('RBFSVM', svm.SVC(kernel='rbf', C=1.25)),
                    ('LogReg', LogisticRegression(max_iter=200, C=0.015))],
        final_estimator=LogisticRegression(max_iter=200))

    clfs = {"Stacked Linear SVM & LogReg": stack_linear,
            "Stacked RBF & Linear SVM": stack_svm,
            "Stacked RBF & LogReg": stack_rbf,
            "Stacked Linear, RBF & LogReg": stack_all}

    print(stack_rbf.get_params(deep=True).keys())
    gamma = 0.6
    for name, clf in clfs.items():
        acc = []
        fscore = []
        if name == "Stacked RBF & LogReg":
            dim = 125
        else:
            dim = 150
        for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, n_splits):
            if name == "Stacked RBF & LogReg" or name == "Stacked Linear, RBF & LogReg" or name == "Stacked RBF & Linear SVM":
                true_gamma = (gamma / (np.array(Xtrain).var() * len(Xtrain.columns)))
                clf.set_params(RBFSVM__gamma=true_gamma)

            clf.fit(Xtrain, ytrain)
            predicted = clf.predict(Xtest)
            acc.append(accuracy_score(ytest, predicted))
            fscore.append(f1_score(ytest, predicted, average='weighted'))
        eval_dict['clf'].append(name)
        eval_dict['accuracy'].append(np.average(acc))
        eval_dict['fscore'].append(np.average(fscore))

    pd.DataFrame(eval_dict).to_csv("./results/stacking/stacking_compare.csv")


# ---------- Graph for above
def plot_stacking():
    stacking = pd.read_csv("./results/stacking/stacking_compare.csv", index_col=0, header=0)
    plt.rcParams['figure.figsize'] = [10, 7]
    X = np.arange(4)
    ind = X + 0.15
    plt.ylim(0.5, 1)
    bars = plt.bar(X + 0.00, stacking['accuracy'], color='royalblue', width=0.3)
    bars2 = plt.bar(X + 0.3, stacking['fscore'], color='lightcoral', width=0.3)

    print(stacking.head(10))
    autolabel(bars, 4)
    autolabel(bars2, 4)

    plt.ylabel('Evaluation Value', size=12)
    plt.xlabel('Classifier', size=12)
    plt.title(
        'Comparing Stacked Classifiers',
        weight='bold', size=14)
    plt.xticks(ind, stacking['clf'], size=10)
    plt.legend(('Accuracy', 'F1 Score'), shadow=True, title="Evaluation Metric", title_fontsize=12)
    plt.show()


# ---------- Confusion matrix for RBF and Linear SVM stacked classifier
def confusion_matrix_svms(dim, n_splits):
    cm = np.zeros((3, 3))
    for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, n_splits):
        model = StackingClassifier(
        estimators=[('LinearSVM', svm.SVC(kernel='linear', C=0.009)),
                    ('RBFSVM', svm.SVC(kernel='rbf', C=1.25))],
        final_estimator=LogisticRegression(max_iter=200))
        model.fit(Xtrain, ytrain)

        predictions = model.predict(Xtest)

        cm += metrics.confusion_matrix(ytest, predictions, normalize='true')

    cm = cm / n_splits
    np.savetxt("./results/stacking/cm_stacked_svm.csv", cm)
    plot_confusion_matrix(cm, "")


# ---------- Producing Kaggle submission for RBF and Linear SVM
def kaggle_submission(dim):
    split_dir = f"all{dim}"

    Xtrain = pd.read_csv(f"./datasets/computed/{split_dir}/all_train_d2v150.csv", index_col=0)
    Xtest = pd.read_csv(f"./datasets/computed/{split_dir}/all_test_d2v150.csv", index_col=0)
    ytrain = pd.read_csv(f"./datasets/computed/{split_dir}/all_train_class.csv", delimiter=',', index_col=0, header=None, names=['rating'])

    print(Xtrain.head(3))
    print(ytrain.head(3))
    print(Xtrain.shape)
    print(ytrain.shape)
    model = StackingClassifier(
        estimators=[('LinearSVM', svm.SVC(kernel='linear', C=0.009)),
                    ('RBFSVM', svm.SVC(kernel='rbf', C=1.25))],
        final_estimator=LogisticRegression(max_iter=200))
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)
    pd.Series(predictions, index=pd.RangeIndex(1, 7019), name='rating').to_csv("results/kaggle/stacked.csv")


if __name__ == "__main__":
    # stacking()
    plot_stacking()
    # confusion_matrix_svms(150, 5)
    # kaggle_submission(150)
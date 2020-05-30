from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

import numpy as np
import scipy
import pandas as pd

from svm import learning_curve
from generate_docvecs import get_doc2vec_crossval
from plotting import plot_confusion_matrix

RANDOM_STATE = 7
CV = 10

# This file contains all the code used in relation to Multinomial Logistic Regression

def autolabel(rects, decimals):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{:.{}f}'.format(height, decimals),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')


# ---------- Logistic Regression for Doc2Vec200 and Doc2Vec100 with PCA(50) vs. Doc2Vec50
# Asking: Do higher dimensionality Doc2Vec text features perform better when reduced to the same dimensionality
# of smaller Doc2Vec features?
def dimensionality_comparison(datasets_dict_noCV, class_labels, n_CV_splits):
    avg_200_acc = []
    avg_200_f1 = []
    avg_100_acc = []
    avg_100_f1 = []
    avg_50_acc = []
    avg_50_f1 = []

    d2v50 = datasets_dict_noCV["Doc2Vec50"]
    d2v100 = datasets_dict_noCV["Doc2Vec100"]
    d2v200 = datasets_dict_noCV["Doc2Vec200"]

    # Run across multiple random states
    for ran_state in range(10):

        ### Doc2Vec200 with PCA = 50
        kf = StratifiedKFold(n_splits=n_CV_splits, shuffle=True, random_state=ran_state)
        pca = PCA(n_components=50, random_state=ran_state)
        lgr = LogisticRegression()

        acc_200 = []
        f1_200 = []
        for train, test in kf.split(d2v200, class_labels):
            d2v200_reduced = pca.fit_transform(d2v200.loc[train])
            lgr.fit(d2v200_reduced, class_labels.loc[train])
            test_reduced = pca.transform(d2v200.loc[test])
            predicted = lgr.predict(test_reduced)
            acc_200.append(lgr.score(test_reduced, class_labels.loc[test]))
            f1_200.append(f1_score(class_labels.loc[test], predicted, average='weighted'))
        avg_200_acc.append(np.average(acc_200))
        avg_200_f1.append(np.average(f1_200))

        ### Doc2Vec100 with PCA = 50
        acc_100 = []
        f1_100 = []
        for train, test in kf.split(d2v100, class_labels):
            d2v100_reduced = pca.fit_transform(d2v100.loc[train])
            lgr.fit(d2v100_reduced, class_labels.loc[train])
            test_reduced = pca.transform(d2v100.loc[test])
            predicted = lgr.predict(test_reduced)
            acc_100.append(lgr.score(test_reduced, class_labels.loc[test]))
            f1_100.append(f1_score(class_labels.loc[test], predicted, average='weighted'))
        avg_100_acc.append(np.average(acc_100))
        avg_100_f1.append(np.average(f1_100))

        ## Doc2Vec50 no PCA
        acc_50 = []
        f1_50 = []
        for train, test in kf.split(d2v50, class_labels):
            lgr.fit(d2v50.loc[train], class_labels.loc[train])
            acc_50.append(lgr.score(d2v50.loc[test], class_labels.loc[test]))
            predicted = lgr.predict(d2v50.loc[test])
            f1_50.append(f1_score(class_labels.loc[test], predicted, average='weighted'))
        avg_50_acc.append(np.average(acc_50))
        avg_50_f1.append(np.average(f1_50))

    results_acc = np.array((np.average(avg_200_acc), np.average(avg_100_acc), np.average(avg_50_acc)))
    results_f1 = np.array((np.average(avg_200_f1), np.average(avg_100_f1), np.average(avg_50_f1)))
    np.save('./results/lgr/lgr_for_pca50_vs_d2v50_acc.npy', results_acc)
    np.save('./results/lgr/lgr_for_pca50_vs_d2v50_f1.npy', results_f1)


# ---------- Graph for above
def plot_dimensionality_comparison():
    results_acc = np.load('./results/lgr/lgr_for_pca50_vs_d2v50_acc.npy')
    results_f1 = np.load('./results/lgr/lgr_for_pca50_vs_d2v50_f1.npy')
    plt.rcParams['figure.figsize'] = [10, 7]
    X = np.arange(3)
    ind = X + 0.15
    plt.ylim(0.8, 0.825)
    bars = plt.bar(X + 0.00, results_acc, color='royalblue', width=0.3)
    bars2 = plt.bar(X + 0.3, results_f1, color='lightcoral', width=0.3)

    autolabel(bars, 4)
    autolabel(bars2, 4)

    plt.ylabel('Evaluation Value', size=12)
    plt.xlabel('Text Features', size=12)
    plt.title(
        'Logistic Regression Accuracy for High Dimension Doc2Vec \n reduced using PCA vs. Lower Dimension Doc2Vec',
        weight='bold', size=14)
    plt.xticks(ind, ('Doc2Vec200 PCA(n=50)',
                     'Doc2Vec100 PCA(n=50)',
                     'Doc2Vec50'), size=10)
    plt.legend(('Accuracy', 'F1 Score'), shadow=True, title="Evaluation Metric", title_fontsize=12)
    plt.show()


# ---------- Selecting K-best CountVec features and adding them to Doc2vec50
# Asking: Does Doc2Vec encompass the same information CountVec does, or can the best features from CountVec
# add additional info (and therefore predictive capability) to a model?
def lgr_kbest_countvec_to_d2v(count_vec, d2v50, class_labels, n_CV_splits, feature_names, K=100):
    kf = StratifiedKFold(n_splits=n_CV_splits, shuffle=True, random_state=RANDOM_STATE)
    lgr = LogisticRegression(max_iter=200)
    avg_acc = []
    avg_f1 = []
    acc = []
    f1_scores = []

    # First with no count_vec attributes
    for train, test in kf.split(d2v50, class_labels):
        lgr.fit(d2v50.loc[train], class_labels.loc[train])
        acc.append(lgr.score(d2v50.loc[test], class_labels.loc[test]))
        predicted = lgr.predict(d2v50.loc[test])
        f1_scores.append(f1_score(class_labels.loc[test], predicted, average='weighted'))
    avg_acc.append(np.average(acc))
    avg_f1.append(np.average(f1_scores))

    # Rest
    for k in range(1, K):
        # Have to regen all best features each time, as the ordering of cols is not the
        # same (even though the first k-1 features will be)
        d2v50_copy = d2v50.copy()
        acc = []
        f1_scores = []
        kbest = SelectKBest(chi2, k=k)
        count_vec_best = kbest.fit_transform(count_vec, class_labels)
        cols = kbest.get_support(indices=True)
        i = 0
        for col in cols:
            # The names of the words will not be correct. The ordering of 'cols' doesnt correspond to count_vec_best
            # The only way to do this would be to store the previous cols result and determine which new
            # value has been added. This is done in the function below but is not necessary here. New columns just
            # need any name
            d2v50_copy[feature_names[col]] = pd.arrays.SparseArray(count_vec_best[:, i].toarray().ravel(), fill_value=0)
            i += 1

        for train, test in kf.split(d2v50_copy, class_labels):
            lgr.fit(d2v50_copy.loc[train], class_labels.loc[train])
            acc.append(lgr.score(d2v50_copy.loc[test], class_labels.loc[test]))
            predicted = lgr.predict(d2v50_copy.loc[test])
            f1_scores.append(f1_score(class_labels.loc[test], predicted, average='weighted'))
        avg_acc.append(np.average(acc))
        avg_f1.append(np.average(f1_scores))
    kbest_results_acc = np.array(avg_acc)
    kbest_results_f1 = np.array(avg_f1)
    np.save('./results/lgr/lgr_for_d2v50_vs_kbest_acc.npy', kbest_results_acc)
    np.save('./results/lgr/lgr_for_d2v50_vs_kbest_f1.npy', kbest_results_f1)

# ---------- Graph for above
def plot_lgr_kbest_countvec_to_d2v(K=30):
    kb_results_acc = np.load('results/lgr/lgr_for_d2v50_vs_kbest_acc.npy')
    kb_results_f1 = np.load('results/lgr/lgr_for_d2v50_vs_kbest_f1.npy')
    plt.rcParams['figure.figsize'] = [10, 7]
    plt.ylim(0.5, 1)
    plt.plot(range(0, K), kb_results_acc, 'bo', linestyle='-', label="Accuracy")
    plt.plot(range(0, K), kb_results_f1, 'ro', linestyle='-', label="F1 Score")
    plt.legend(loc='upper left', shadow=True, title="Evaluation Metric", title_fontsize=12)
    plt.xlabel("Number of K-best Count Vec Features Included", size=12)
    plt.ylabel("Evaluation Value", size=12)
    plt.title("Logistic Regression Accuracy for Doc2Vec50 vs. \n Number of included K-best CountVectorizer Features",
              weight="bold", size=14)
    plt.show()


# ---------- Finding top K words by Chi^2 Score
# Inspecting the data to determine if the relationships between words are intuitive
def plot_kbest_words(count_vec, class_labels, K=20):
    cols = set()
    top_names = []
    occurences = []
    for i in range(1, K + 1):
        kbest = SelectKBest(chi2, k=i)
        count_vec_best = kbest.fit_transform(count_vec, class_labels)
        cols_curr = set(kbest.get_support(indices=True))
        new_idx = list(cols_curr - cols).pop()
        top_names.append(feature_names[new_idx])
        occurences.append(np.sum(count_vec[:, new_idx].toarray().ravel()))
        cols = cols_curr

    plt.rcParams['figure.figsize'] = [10, 7]
    X = np.arange(K)
    plt.subplot(2, 1, 1)
    bars = plt.bar(X, occurences, color=sns.color_palette("GnBu_d", K), width=0.50)

    autolabel(bars, 0)
    plt.ylabel('Occurences', size=12)
    plt.ylim(0, 18500)
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    plt.title(f'Occurences of top {K} CountVectoriser Words Ordered by Chi^2 Score \n', weight='bold', size=14)
    # plt.xticks(ind, top_names, rotation=70, size=10)

    plt.rcParams['figure.figsize'] = [10, 7]
    X = np.arange(K)
    ind = X
    plt.subplot(2, 1, 2)
    bars = plt.bar(X, sorted(list(kbest.scores_), reverse=True)[0:K], color=sns.color_palette("GnBu_d", K), width=0.50)

    autolabel(bars, 0)
    plt.ylabel('Chi^2 Score', size=12)
    plt.xlabel('Word', size=12)
    plt.ylim(0, 2750)
    plt.xticks(ind, top_names, rotation=70, size=10)
    plt.tight_layout()
    plt.show()


# ---------- Logisitc Regression with CountVec Kbest (No Doc2Vec)
# Asking: How well do the top K words predict ratings themselves, without the more sophisticated
# Doc2Vec features?
def lgr_kbest_countvec(count_vec, class_labels, n_CV_splits, K=100):
    kf = StratifiedKFold(n_splits=n_CV_splits, shuffle=True, random_state=RANDOM_STATE)
    lgr = LogisticRegression(max_iter=200)
    avg_acc = []
    avg_f1 = []
    # Now all best features from 1 to 30
    for k in range(1, K + 1):
        # Have to regen all best features each time, as the ordering of cols is not the
        # same (even though the first k-1 features will be)
        acc = []
        f1_scores = []
        kbest = SelectKBest(chi2, k=k)
        count_vec_best = kbest.fit_transform(count_vec, class_labels)
        df = pd.DataFrame.sparse.from_spmatrix(count_vec_best)
        for train, test in kf.split(df, class_labels):
            lgr.fit(df.loc[train], class_labels.loc[train])
            acc.append(lgr.score(df.loc[test], class_labels.loc[test]))
            predicted = lgr.predict(df.loc[test])
            f1_scores.append(f1_score(class_labels.loc[test], predicted, average='weighted'))
        avg_acc.append(np.average(acc))
        avg_f1.append(np.average(f1_scores))
    kbest_results_acc = np.array(avg_acc)
    kbest_results_f1 = np.array(avg_f1)
    np.save('./results/lgr/lgr_for_countvec_vs_kbest_acc.npy', kbest_results_acc)
    np.save('./results/lgr/lgr_for_countvec_vs_kbest_f1.npy', kbest_results_f1)


# ---------- Graph for above
def plot_lgr_kbest_countvec(K=100):
    kb_results_acc = np.load('./results/lgr/lgr_for_countvec_vs_kbest_acc.npy')
    kb_results_f1 = np.load('./results/lgr/lgr_for_countvec_vs_kbest_f1.npy')
    plt.rcParams['figure.figsize'] = [10, 7]
    plt.ylim(0.5, 1)
    plt.plot(range(0, K), kb_results_acc, 'bo', linestyle='-', label="Accuracy")
    plt.plot(range(0, K), kb_results_f1, 'ro', linestyle='-', label="F1 Score")
    plt.legend(loc='upper left', shadow=True, title="Evaluation Metric", title_fontsize=12)
    plt.xlabel("K-best Features", size=12)
    plt.ylabel("Evaluation Value", size=12)
    plt.title("Logistic Regression Accuracy for CountVec vs. \n Number K-best Features", weight="bold", size=14)
    plt.show()


# ---------- Plotting lgr_kbest_countvec & lgr_kbest_countvec_to_d2v together
def plot_lgr_countvec_vs_d2v():
    kb_results_acc = np.load('results/lgr/lgr_for_d2v50_vs_kbest_acc.npy')
    kb_results_f1 = np.load('results/lgr/lgr_for_d2v50_vs_kbest_f1.npy')

    plt.rcParams['figure.figsize'] = [10, 7]
    plt.ylim(0.5, 1)

    plt.plot(range(0, 30), kb_results_acc, 'bo', linestyle='-', label="D2V + CountVev Accuracy")
    plt.plot(range(0, 30), kb_results_f1, 'ro', linestyle='-', label="D2V + CountVev F1 Score")

    kb_results_acc = np.load('./results/lgr/lgr_for_countvec_vs_kbest_acc.npy')
    kb_results_f1 = np.load('./results/lgr/lgr_for_countvec_vs_kbest_f1.npy')

    plt.plot(range(0, 100), kb_results_acc, 'go', linestyle='-', label="CountVec Accuracy")
    plt.plot(range(0, 100), kb_results_f1, 'yo', linestyle='-', label="CountVec F1 Score")

    plt.legend(loc='upper left', shadow=True)
    plt.xlabel("Number of K-best Count Vec Features Included", size=12)
    plt.ylabel("Evaluation Value", size=12)
    plt.title("Logistic Regression Accuracy for Doc2Vec50 including K-best \n CountVectorizer Features vs. CountVectorizer Features Alone",
              weight="bold", size=14)
    plt.show()

# ---------- Learning Curves for Logisitc Regression vs Dimensionality of Doc2Vec
def lgr_learning_curve():
    lgr = LogisticRegression()
    train, test = learning_curve(lgr)
    train.to_csv("./results/lgr/learning_curve_train.csv")
    test.to_csv("./results/lgr/learning_curve_test.csv")


# ---------- Graph for Above
def plot_lgr_learning_curve():

    train = pd.read_csv("./results/lgr/learning_curve_train.csv", index_col=0, header=0)
    test = pd.read_csv("./results/lgr/learning_curve_test.csv", index_col=0, header=0)

    plt.rcParams['figure.figsize'] = [10, 7]
    plt.ylim(0.75, 0.9)

    plt.plot(train['dim'], train['fscore'], 'go', linestyle='-', label="Train F1 Score")
    plt.plot(train['dim'], train['accuracy'], 'yo', linestyle='-', label="Train Accuracy")

    plt.plot(test['dim'], test['fscore'], 'bo', linestyle='-', label="Test F1 Score")
    plt.plot(test['dim'], test['accuracy'], 'ro', linestyle='-', label="Test Accuracy")

    plt.legend(loc='upper left', shadow=True)
    plt.xlabel("Doc2Vec Dimensionality", size=12)
    plt.ylabel("Evaluation Value", size=12)
    plt.title("Learning Curves for Doc2Vec Logisitc Regression vs. Dimensionality",
              weight="bold", size=14)
    plt.show()


# ---------- AdaBoost Logistic Regression, Logisitic Regression, and AdaB Decision Tree
def ensemble_compare(dim, n_CV_splits):
    eval_dict = {
        "clf": [],
        "fscore": [],
        "accuracy": []
    }

    ab_lgr = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=200, C=0.015), random_state=RANDOM_STATE)
    ab_dt = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), random_state=RANDOM_STATE)
    lgr = LogisticRegression(max_iter=200, random_state=RANDOM_STATE, C=0.015)
    bc_lgr = BaggingClassifier(base_estimator=LogisticRegression(max_iter=200, C=0.015), random_state=RANDOM_STATE)

    clfs = {"Adaboost LogReg": ab_lgr, "Adaboost Decision Tree": ab_dt, "Logisitc Regression": lgr, "Bagging LogReg": bc_lgr}

    for name, clf in clfs.items():
        acc = []
        fscore = []
        i=0
        for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, n_CV_splits):
            clf.fit(Xtrain, ytrain)
            acc.append(clf.score(Xtest, ytest))
            predicted = clf.predict(Xtest)
            fscore.append(f1_score(ytest, predicted, average='weighted'))
        # Assigned in wrong order, accounted for in plot
        eval_dict['clf'].append(name)
        eval_dict['accuracy'].append(np.average(fscore))
        eval_dict['fscore'].append(np.average(acc))

    pd.DataFrame(eval_dict).to_csv("./results/lgr/ensemble_compare.csv")


# ---------- Graph for Above
def plot_ensemble_compare():
    ensemble = pd.read_csv("./results/lgr/ensemble_compare.csv", index_col=0, header=0)
    plt.rcParams['figure.figsize'] = [10, 7]
    X = np.arange(4)
    ind = X + 0.15
    plt.ylim(0.5, 1)
    bars = plt.bar(X + 0.00, ensemble['accuracy'], color='royalblue', width=0.3)
    bars2 = plt.bar(X + 0.3, ensemble['fscore'], color='lightcoral', width=0.3)

    print(ensemble.head(10))
    autolabel(bars, 4)
    autolabel(bars2, 4)

    plt.ylabel('Evaluation Value', size=12)
    plt.xlabel('Classifier', size=12)
    plt.title(
        'Comparing Ensemble Classifiers for Doc2Vec150',
        weight='bold', size=14)
    plt.xticks(ind, ensemble['clf'], size=10)
    plt.legend(('F1 Score', 'Accuracy'), shadow=True, title="Evaluation Metric", title_fontsize=12)
    plt.show()


# ---------- Bagging Critical Analysis
def bagging(dim, n_CV_splits):
    eval_dict = {
        "clf": [],
        "fscore": [],
        "accuracy": []
    }
    acc_bc = []
    acc_lgr = []
    fscore_bc = []
    fscore_lgr = []
    for ran_state in range(10):
        lgr = LogisticRegression(max_iter=200, random_state=ran_state, C=0.015)
        bc_lgr = BaggingClassifier(base_estimator=LogisticRegression(max_iter=200, C=0.015), random_state=ran_state)

        clfs = {"Logisitc Regression": lgr,
                "Bagging LogReg": bc_lgr}
        for name, clf in clfs.items():
            acc = []
            fscore = []
            for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, n_CV_splits):
                clf.fit(Xtrain, ytrain)
                acc.append(clf.score(Xtest, ytest))
                predicted = clf.predict(Xtest)
                fscore.append(f1_score(ytest, predicted, average='weighted'))
            if name == "Logisitc Regression":
                acc_lgr.append(np.average(acc))
                fscore_lgr.append(np.average(fscore))
            else:
                acc_bc.append(np.average(acc))
                fscore_bc.append(np.average(fscore))

    # Wrong order for accuracy and fscore, accounted for in plot
    eval_dict['clf'].append("Logisitc Regression")
    eval_dict['accuracy'].append(np.average(fscore_lgr))
    eval_dict['fscore'].append(np.average(acc_lgr))
    eval_dict['clf'].append("Bagging LogReg")
    eval_dict['accuracy'].append(np.average(fscore_bc))
    eval_dict['fscore'].append(np.average(acc_bc))

    pd.DataFrame(eval_dict).to_csv("./results/lgr/bagging_compare.csv")


# ---------- Graph for above
def plot_bagging():
    bagging = pd.read_csv("./results/lgr/bagging_compare.csv", index_col=0, header=0)
    plt.rcParams['figure.figsize'] = [10, 7]
    X = np.arange(2)
    ind = X + 0.15
    plt.ylim(0.5, 1)
    bars = plt.bar(X + 0.00, bagging['accuracy'], color='royalblue', width=0.3)
    bars2 = plt.bar(X + 0.3, bagging['fscore'], color='lightcoral', width=0.3)

    print(bagging.head(10))
    autolabel(bars, 4)
    autolabel(bars2, 4)

    plt.ylabel('Evaluation Value', size=12)
    plt.xlabel('Classifier', size=12)
    plt.title(
        'Bagging Logisitic Regression vs Logisitic Regression for Doc2Vec150',
        weight='bold', size=14)
    plt.xticks(ind, bagging['clf'], size=10)
    plt.legend(('F1 Score', 'Accuracy'), shadow=True, title="Evaluation Metric", title_fontsize=12)
    plt.show()


# ---------- Adaboost Critical Analysis
def adaboost(dim, n_CV_splits):
    eval_dict = {
        "clf": [],
        "fscore": [],
        "accuracy": []
    }

    lgr = LogisticRegression(max_iter=200, random_state=RANDOM_STATE, C=0.015)
    ab_lgr = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=200, C=0.31), random_state=RANDOM_STATE)

    clfs = {"Logisitc Regression": lgr,
            "AdaBoost LogReg": ab_lgr}
    for name, clf in clfs.items():
        acc = []
        fscore = []
        for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, n_CV_splits):
            clf.fit(Xtrain, ytrain)
            acc.append(clf.score(Xtest, ytest))
            predicted = clf.predict(Xtest)
            fscore.append(f1_score(ytest, predicted, average='weighted'))

        eval_dict['clf'].append(name)
        eval_dict['accuracy'].append(np.average(acc))
        eval_dict['fscore'].append(np.average(fscore))

    pd.DataFrame(eval_dict).to_csv("./results/lgr/adaboost_compare.csv")


# ---------- Graph for above
def plot_adaboost():
    bagging = pd.read_csv("./results/lgr/adaboost_compare.csv", index_col=0, header=0)
    plt.rcParams['figure.figsize'] = [10, 7]
    X = np.arange(2)
    ind = X + 0.15
    plt.ylim(0.5, 1)
    bars = plt.bar(X + 0.00, bagging['accuracy'], color='royalblue', width=0.3)
    bars2 = plt.bar(X + 0.3, bagging['fscore'], color='lightcoral', width=0.3)

    print(bagging.head(10))
    autolabel(bars, 4)
    autolabel(bars2, 4)

    plt.ylabel('Evaluation Value', size=12)
    plt.xlabel('Classifier', size=12)
    plt.title(
        'Adaboost Logistic Regression vs Logistic Regression for Doc2Vec150',
        weight='bold', size=14)
    plt.xticks(ind, bagging['clf'], size=10)
    plt.legend(('Accuracy', 'F1 Score'), shadow=True, title="Evaluation Metric", title_fontsize=12)
    plt.show()


# ---------- GridSearch C Hyperparameter
def gridsearch_c(param_space, dim, xval_size=5):
    for i, c in enumerate(param_space):

        print(f"({i}/{len(param_space)}) values searched (C={c}).")

        model = LogisticRegression(max_iter=200, C=c)

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

        with open(f"./results/lgr/gridsearch_{dim}.csv", "a") as fp:
            fp.write(
                f"{c}, {fscore / xval_size}, {accuracy / xval_size}, {precision / xval_size}, {recall / xval_size}\n")


# ---------- Graph for above
def plot_gridsearch_c(dim):
    gridsearch = pd.read_csv(f"./results/lgr/gridsearch_{dim}.csv", sep=', ', names=['C', 'fscore', 'accuracy','precision','recall'])
    print(gridsearch.head(10))
    plt.rcParams['figure.figsize'] = [10, 7]
    plt.xscale('log')
    plt.ylim(0.75, 0.9)

    plt.plot(gridsearch['C'], gridsearch['fscore'], 'go', linestyle='-', label="F1 Score")
    plt.plot(gridsearch['C'], gridsearch['accuracy'], 'yo', linestyle='-', label="Accuracy")


    plt.legend(loc='upper left', shadow=True)
    plt.xlabel("C Hyperparameter", size=12)
    plt.ylabel("Evaluation Value", size=12)
    plt.title("Logistic Regression for Doc2Vec150 vs C Hyperparameter",
              weight="bold", size=14)
    plt.show()


# ---------- Confusion Matrix for Doc2Vec150
def confusion_matrix(dim, n_splits):
    cm = np.zeros((3, 3))
    for Xtrain, Xtest, ytrain, ytest in get_doc2vec_crossval(dim, n_splits):
        lgr = LogisticRegression(max_iter=200, C=0.015)
        lgr.fit(Xtrain, ytrain)

        predictions = lgr.predict(Xtest)

        cm += metrics.confusion_matrix(ytest, predictions, normalize='true')

    cm = cm / n_splits
    np.savetxt("./results/lgr/cm_final.csv", cm)
    plot_confusion_matrix(cm, "Logistic Regression Confusion Matrix (Doc2Vec150)")


# ---------- Producing Kaggle submission for Logistic Regression
def kaggle_submission(dim):
    split_dir = f"all{dim}"

    Xtrain = pd.read_csv(f"./datasets/computed/{split_dir}/all_train_d2v150.csv", index_col=0)
    Xtest = pd.read_csv(f"./datasets/computed/{split_dir}/all_test_d2v150.csv", index_col=0)
    ytrain = pd.read_csv(f"./datasets/computed/{split_dir}/all_train_class.csv", delimiter=',', index_col=0, header=None, names=['rating'])

    print(Xtrain.head(3))
    print(ytrain.head(3))
    print(Xtrain.shape)
    print(ytrain.shape)
    model = LogisticRegression(max_iter=200, C=0.015)
    model.fit(Xtrain, ytrain)
    predictions = model.predict(Xtest)
    pd.Series(predictions, index=pd.RangeIndex(1, 7019), name='rating').to_csv("./results/kaggle/lgr.csv")




if __name__ == "__main__":

    # ---------- Loading in all data
    # Meta adaboosting data
    train_set = pd.read_csv(r"./datasets/review_meta_train.csv")
    class_labels = train_set['rating']

    # Count vec
    vocab = pickle.load(open("./datasets/review_text_features_countvec/train_countvectorizer.pkl", "rb"))
    vocab_dict = vocab.vocabulary_
    count_vec = scipy.sparse.load_npz('./datasets/review_text_features_countvec/review_text_train_vec.npz')
    feature_names = list(vocab.get_feature_names())

    # doc2vec50
    d2v50 = pd.read_csv(r"./datasets/review_text_features_doc2vec50/review_text_train_doc2vec50.csv", index_col=False,
                        delimiter=',', header=None)

    # doc2vec100
    d2v100 = pd.read_csv(r"./datasets/review_text_features_doc2vec100/review_text_train_doc2vec100.csv",
                         index_col=False, delimiter=',', header=None)

    # doc2vec200
    d2v200 = pd.read_csv(r"./datasets/review_text_features_doc2vec200/review_text_train_doc2vec200.csv",
                         index_col=False, delimiter=',', header=None)

    datasets_dict = {"Count Vectoriser": count_vec, "Doc2Vec50": d2v50, "Doc2Vec100": d2v100, "Doc2Vec200": d2v200}
    datasets_dict_noCV = {"Doc2Vec50": d2v50, "Doc2Vec100": d2v100, "Doc2Vec200": d2v200}

    # ---------- All functions

    n_CV_splits = 10

    # dimensionality_comparison(datasets_dict_noCV, class_labels, n_CV_splits)
    # plot_dimensionality_comparison()

    # lgr_C(d2v50, class_labels, n_CV_splits, c_range=np.logspace(-4, 4, 30))
    # plot_lgr_C(c_range=np.logspace(-4, 4, 30))
    #
    # lgr_kbest_countvec_to_d2v(count_vec, d2v50, class_labels, n_CV_splits, feature_names, K=30)
    # plot_lgr_kbest_countvec_to_d2v(K=30)
    #
    # plot_kbest_words(count_vec, class_labels, K=20)
    #
    # lgr_kbest_countvec(count_vec, class_labels, n_CV_splits, K=100)
    # plot_lgr_kbest_countvec(K=100)

    # plot_lgr_countvec_vs_d2v()
    #
    # lgr_learning_curve()
    # plot_lgr_learning_curve()

    # ensemble_compare(150, 5)
    # plot_ensemble_compare()

    # bagging(150, 5)
    # plot_bagging()

    # adaboost(150, 5)
    # plot_adaboost()

    # param_space = [0.0001, 0.001, 0.01, 0.015, 0.02, 0.05, 0.08, 0.1, 1, 10]
    # gridsearch_c(param_space, 150, xval_size=5)
    # plot_gridsearch_c(150)

    # confusion_matrix(150, 5)
    # kaggle_submission(150)
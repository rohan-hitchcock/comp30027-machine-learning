from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 7]

import numpy as np
import scipy
import pandas as pd

RANDOM_STATE = 7
CV = 10

# Meta training data
train = pd.read_csv(r"./datasets/review_meta_train.csv")
test = pd.read_csv(r"./datasets/review_meta_test.csv")

# Label indices
low_rating = train.loc[train['rating'] == 1].index
med_rating = train.loc[train['rating'] == 3].index
high_rating = train.loc[train['rating'] == 5].index

# Count vec
vocab = pickle.load(open("./datasets/review_text_features_countvec/train_countvectorizer.pkl", "rb"))
vocab_dict = vocab.vocabulary_
count_vec = scipy.sparse.load_npz('./datasets/review_text_features_countvec/review_text_train_vec.npz')
feature_names = list(vocab.get_feature_names())

# doc2vec50
d2v50 = pd.read_csv(r"./datasets/review_text_features_doc2vec50/review_text_train_doc2vec50.csv", index_col=False,
                    delimiter=',', header=None)
# print(d2v50.head(10))

# doc2vec100
d2v100 = pd.read_csv(r"./datasets/review_text_features_doc2vec100/review_text_train_doc2vec100.csv", index_col=False,
                     delimiter=',', header=None)
# print(d2v100.head(10))

# doc2vec200
d2v200 = pd.read_csv(r"./datasets/review_text_features_doc2vec200/review_text_train_doc2vec200.csv", index_col=False,
                     delimiter=',', header=None)
# print(d2v200.head(10))

datasets_dict = {"Count Vectoriser": count_vec, "Doc2Vec50": d2v50, "Doc2Vec100": d2v100, "Doc2Vec200": d2v200}
y_train = train['rating']
# Meta without any of the non-numeric attributes
x_train = train.drop(['rating', 'date', 'review_id', 'reviewer_id', 'business_id'], axis=1)

# -------------- Logistic Regression for Doc2Vec200 and Doc2Vec100 with PCA(50) vs. Doc2Vec50
# Asking: Do higher dimensionality Doc2Vec text features perform better when reduced to the same dimensionality
# of smaller Doc2Vec features?
#
# avg_200_acc = []
# avg_200_f1 = []
# avg_100_acc = []
# avg_100_f1 = []
# avg_50_acc = []
# avg_50_f1 = []
#
# #Run across multiple random states
# for ran_state in range(10):
#
#     ### Doc2Vec200 with PCA = 50
#     kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=ran_state)
#     pca = PCA(n_components=50, random_state=ran_state)
#     lgr = LogisticRegression()
#
#     acc_200 = []
#     f1_200 = []
#     for train, test in kf.split(d2v200, y_train):
#         d2v200_reduced = pca.fit_transform(d2v200.loc[train])
#         lgr.fit(d2v200_reduced, y_train.loc[train])
#         test_reduced = pca.transform(d2v200.loc[test])
#         predicted = lgr.predict(test_reduced)
#         acc_200.append(lgr.score(test_reduced, y_train.loc[test]))
#         f1_200.append(f1_score(y_train.loc[test], predicted, average='weighted'))
#     avg_200_acc.append(np.average(acc_200))
#     avg_200_f1.append(np.average(f1_200))
#
#     ### Doc2Vec100 with PCA = 50
#     acc_100 = []
#     f1_100 = []
#     for train, test in kf.split(d2v100, y_train):
#         d2v100_reduced = pca.fit_transform(d2v100.loc[train])
#         lgr.fit(d2v100_reduced, y_train.loc[train])
#         test_reduced = pca.transform(d2v100.loc[test])
#         predicted = lgr.predict(test_reduced)
#         acc_100.append(lgr.score(test_reduced, y_train.loc[test]))
#         f1_100.append(f1_score(y_train.loc[test], predicted, average='weighted'))
#     avg_100_acc.append(np.average(acc_100))
#     avg_100_f1.append(np.average(f1_100))
#
#     ## Doc2Vec50 no PCA
#     acc_50 = []
#     f1_50 = []
#     for train, test in kf.split(d2v50, y_train):
#         lgr.fit(d2v50.loc[train], y_train.loc[train])
#         acc_50.append(lgr.score(d2v50.loc[test], y_train.loc[test]))
#         predicted = lgr.predict(d2v50.loc[test])
#         f1_50.append(f1_score(y_train.loc[test], predicted, average='weighted'))
#     avg_50_acc.append(np.average(acc_50))
#     avg_50_f1.append(np.average(f1_50))
#
# # print(f"Average Accuracy for Logistic Regression with Cross Validation (10), PCA(50) for Doc2Vec200: {np.average(avg_200)}")
# # print(f"Average Accuracy for Logistic Regression with Cross Validation (10), PCA(50) for Doc2Vec100: {np.average(avg_100)}")
# # print(f"Average Accuracy for Logistic Regression with Cross Validation (10) for Doc2Vec50: {np.average(avg_50)}")
#
# results_acc = np.array((np.average(avg_200_acc), np.average(avg_100_acc), np.average(avg_50_acc)))
# results_f1 = np.array((np.average(avg_200_f1), np.average(avg_100_f1), np.average(avg_50_f1)))
# np.save('./results/lgr_for_pca50_vs_d2v50_acc.npy', results_acc)
# np.save('./results/lgr_for_pca50_vs_d2v50_f1.npy', results_f1)


# ------------- Graph for above
# Can simply load to produce graph
# results_acc = np.load('./results/lgr_for_pca50_vs_d2v50_acc.npy')
# results_f1 = np.load('./results/lgr_for_pca50_vs_d2v50_f1.npy')
# plt.rcParams['figure.figsize'] = [10, 7]
# X = np.arange(3)
# ind = X + 0.15
# # fig = plt.figure()
# # ax = fig.add_axes([0, 0, 1, 1])
# plt.ylim(0.8, 0.825)
# bars = plt.bar(X + 0.00, results_acc, color = 'royalblue', width=0.3)
# bars2 = plt.bar(X + 0.3, results_f1, color = 'lightcoral', width=0.3)
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         plt.annotate('{:.4f}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(bars)
# autolabel(bars2)
# plt.ylabel('Evaluation Value', size=12)
# plt.xlabel('Text Features', size=12)
# plt.title('Logistic Regression Accuracy for High Dimension Doc2Vec \n reduced using PCA vs. Lower Dimension Doc2Vec', weight='bold',size=14)
# plt.xticks(ind, ('Doc2Vec200 PCA(n=50)',
#                  'Doc2Vec100 PCA(n=50)',
#                  'Doc2Vec50'), size=10)
# plt.legend(('Accuracy', 'F1 Score'), shadow=True, title="Evaluation Metric", title_fontsize=12)
# plt.show()

# ----- Varying C parameter of logistic Regression
#
# kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
# c_acc = []
# c_f1_scores = []
# for c in np.logspace(-4, 4, 30):
#     lgr = LogisticRegression(C=c)
#     acc_50 = []
#     f1_scores_50 = []
#     for train, test in kf.split(d2v50, y_train):
#         lgr.fit(d2v50.loc[train], y_train.loc[train])
#         acc_50.append(lgr.score(d2v50.loc[test], y_train.loc[test]))
#         predicted = lgr.predict(d2v50.loc[test])
#         f1_scores_50.append(f1_score(y_train.loc[test], predicted, average='weighted'))
#     c_acc.append(np.average(acc_50))
#     c_f1_scores.append(np.average(f1_scores_50))
#
#
# c_results_acc = np.array(c_acc)
# c_results_f1 = np.array(c_f1_scores)
# np.save('./results/lgr_for_d2v50_vs_C_acc.npy', c_results_acc)
# np.save('./results/lgr_for_d2v50_vs_C_f1.npy', c_results_f1)
#
# # Can simply load to produce graph
# c_results_acc = np.load('./results/lgr_for_d2v50_vs_C_acc.npy')
# c_results_f1 = np.load('./results/lgr_for_d2v50_vs_C_f1.npy')
# plt.rcParams['figure.figsize'] = [10, 7]
# plt.xscale('log')
# plt.ylim(0.5, 1)
# plt.plot(np.logspace(-4, 4, 30), c_results_acc, 'bo', linestyle='-', label="Accuracy")
# plt.plot(np.logspace(-4, 4, 30), c_results_f1, 'ro', linestyle='-', label="F1 Score")
# plt.legend(loc='upper left', shadow=True, title="Evaluation Metric", title_fontsize=12)
# plt.xlabel("C Value", size=12)
# plt.ylabel("Evaluation Value", size=12)
# plt.title("Logistic Regression Accuracy for Doc2Vec50 vs. C Hyperparameter", weight="bold", size=14)
# plt.show()


# ------- Selecting K-best CountVec features and adding them to Doc2vec50
# First with no count_vec attributes
# kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
# lgr = LogisticRegression(max_iter=200)
# avg_acc = []
# avg_f1 = []
# acc = []
# f1_scores = []
# for train, test in kf.split(d2v50, y_train):
#     lgr.fit(d2v50.loc[train], y_train.loc[train])
#     acc.append(lgr.score(d2v50.loc[test], y_train.loc[test]))
#     predicted = lgr.predict(d2v50.loc[test])
#     f1_scores.append(f1_score(y_train.loc[test], predicted, average='weighted'))
# avg_acc.append(np.average(acc))
# avg_f1.append(np.average(f1_scores))
#
# # Now all best features from 1 to 30
# K = 30
# for k in range(1, K):
#     # Have to regen all best features each time, as the ordering of cols is not the
#     # same (even though the first k-1 features will be)
#     d2v50_copy = d2v50.copy()
#     acc = []
#     f1_scores = []
#     kbest = SelectKBest(chi2, k=k)
#     count_vec_best = kbest.fit_transform(count_vec, y_train)
#     cols = kbest.get_support(indices=True)
#     i = 0
#     for col in cols:
#         d2v50_copy[feature_names[cols[i]]] = pd.arrays.SparseArray(count_vec_best[:, i].toarray().ravel(), fill_value=0)
#         i += 1
#     for train, test in kf.split(d2v50_copy, y_train):
#         lgr.fit(d2v50_copy.loc[train], y_train.loc[train])
#         acc.append(lgr.score(d2v50_copy.loc[test], y_train.loc[test]))
#         predicted = lgr.predict(d2v50_copy.loc[test])
#         f1_scores.append(f1_score(y_train.loc[test], predicted, average='weighted'))
#     avg_acc.append(np.average(acc))
#     avg_f1.append(np.average(f1_scores))
# kbest_results_acc = np.array(avg_acc)
# kbest_results_f1 = np.array(avg_f1)
# np.save('./results/lgr_for_d2v50_vs_kbest_acc.npy', kbest_results_acc)
# np.save('./results/lgr_for_d2v50_vs_kbest_f1.npy', kbest_results_f1)

# Can simply load to produce graph
kb_results_acc = np.load('results/lgr/lgr_for_d2v50_vs_kbest_acc.npy')
kb_results_f1 = np.load('results/lgr/lgr_for_d2v50_vs_kbest_f1.npy')
plt.rcParams['figure.figsize'] = [10, 7]
plt.ylim(0.5, 1)
plt.plot(range(0, 30), kb_results_acc, 'bo', linestyle='-', label="Accuracy")
plt.plot(range(0, 30), kb_results_f1, 'ro', linestyle='-', label="F1 Score")
plt.legend(loc='upper left', shadow=True, title="Evaluation Metric", title_fontsize=12)
plt.xlabel("Number of K-best Count Vec Features Included", size=12)
plt.ylabel("Evaluation Value", size=12)
plt.title("Logistic Regression Accuracy for Doc2Vec50 vs. \n Number of included K-best CountVectorizer Features", weight="bold", size=14)
plt.show()


# ------- Simply finding the top K best words by Chi2
# k=20
# cols=set()
# top_names = []
# occurences = []
# for i in range(1, k+1):
#     kbest = SelectKBest(chi2, k=i)
#     count_vec_best = kbest.fit_transform(count_vec, y_train)
#     cols_curr = set(kbest.get_support(indices=True))
#     new_idx = list(cols_curr-cols).pop()
#     top_names.append(feature_names[new_idx])
#     occurences.append(np.sum(count_vec[:, new_idx].toarray().ravel()))
#     cols = cols_curr
#
# plt.rcParams['figure.figsize'] = [10, 7]
# X = np.arange(k)
# ind = X
# plt.subplot(2, 1, 1)
# bars = plt.bar(X, occurences, color=sns.color_palette("GnBu_d", k), width=0.50)
# print(len(kbest.scores_))
# print(count_vec_best.shape)
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         plt.annotate('{:.0f}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(bars)
# plt.ylabel('Occurences', size=12)
# plt.ylim(0, 18500)
# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False)
# plt.title(f'Occurences of top {k} CountVectoriser Words Ordered by Chi^2 Score \n', weight='bold', size=14)
# # plt.xticks(ind, top_names, rotation=70, size=10)
#
#
# plt.rcParams['figure.figsize'] = [10, 7]
# X = np.arange(k)
# ind = X
# plt.subplot(2, 1, 2)
# bars = plt.bar(X, sorted(list(kbest.scores_), reverse=True)[0:k], color=sns.color_palette("GnBu_d", k), width=0.50)
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         plt.annotate('{:.0f}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(bars)
# plt.ylabel('Chi^2 Score', size=12)
# plt.xlabel('Word', size=12)
# plt.ylim(0, 2750)
# plt.xticks(ind, top_names, rotation=70, size=10)
# plt.tight_layout()
# plt.show()


# ----- Logisitc Regression with CountVec Kbest (No Doc2Vec)
# kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=RANDOM_STATE)
# lgr = LogisticRegression(max_iter=200)
# avg_acc = []
# avg_f1 = []
# acc = []
# f1_scores = []
# # Now all best features from 1 to 30
# K = 100
# for k in range(1, K+1):
#     # Have to regen all best features each time, as the ordering of cols is not the
#     # same (even though the first k-1 features will be)
#     acc = []
#     f1_scores = []
#     kbest = SelectKBest(chi2, k=k)
#     count_vec_best = kbest.fit_transform(count_vec, y_train)
#     df = pd.DataFrame.sparse.from_spmatrix(count_vec_best)
#     for train, test in kf.split(df, y_train):
#         lgr.fit(df.loc[train], y_train.loc[train])
#         acc.append(lgr.score(df.loc[test], y_train.loc[test]))
#         predicted = lgr.predict(df.loc[test])
#         f1_scores.append(f1_score(y_train.loc[test], predicted, average='weighted'))
#     avg_acc.append(np.average(acc))
#     avg_f1.append(np.average(f1_scores))
# kbest_results_acc = np.array(avg_acc)
# kbest_results_f1 = np.array(avg_f1)
# np.save('./results/lgr_for_countvec_vs_kbest_acc.npy', kbest_results_acc)
# np.save('./results/lgr_for_countvec_vs_kbest_f1.npy', kbest_results_f1)
#
# # Can simply load to produce graph
# kb_results_acc = np.load('./results/lgr_for_countvec_vs_kbest_acc.npy')
# kb_results_f1 = np.load('./results/lgr_for_countvec_vs_kbest_f1.npy')
# plt.rcParams['figure.figsize'] = [10, 7]
# plt.ylim(0.5, 1)
# plt.plot(range(0, K), kb_results_acc, 'bo', linestyle='-', label="Accuracy")
# plt.plot(range(0, K), kb_results_f1, 'ro', linestyle='-', label="F1 Score")
# plt.legend(loc='upper left', shadow=True, title="Evaluation Metric", title_fontsize=12)
# plt.xlabel("K-best Features", size=12)
# plt.ylabel("Evaluation Value", size=12)
# plt.title("Logistic Regression Accuracy for CountVec vs. \n Number K-best Features", weight="bold", size=14)
# plt.show()
from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
plt.rcParams['figure.figsize'] = [10, 7]
import numpy as np
import scipy
import pandas as pd

RANDOM_STATE = 7
CV = 10

# Meta training data
train = pd.read_csv(r"./datasets/review_meta_train.csv")
test = pd.read_csv(r"./datasets/review_meta_test.csv")

#Label indices
low_rating = train.loc[train['rating'] == 1].index
med_rating = train.loc[train['rating'] == 3].index
high_rating = train.loc[train['rating'] == 5].index

# Count vec
vocab = pickle.load(open("./datasets/review_text_features_countvec/train_countvectorizer.pkl", "rb"))
vocab_dict = vocab.vocabulary_
count_vec = scipy.sparse.load_npz('./datasets/review_text_features_countvec/review_text_train_vec.npz')

# doc2vec50
d2v50 = pd.read_csv(r"./datasets/review_text_features_doc2vec50/review_text_train_doc2vec50.csv", index_col=False, delimiter=',', header=None)
# print(d2v50.head(10))

# doc2vec100
d2v100 = pd.read_csv(r"./datasets/review_text_features_doc2vec100/review_text_train_doc2vec100.csv", index_col=False, delimiter=',', header=None)
# print(d2v100.head(10))

# doc2vec200
d2v200 = pd.read_csv(r"./datasets/review_text_features_doc2vec200/review_text_train_doc2vec200.csv", index_col=False, delimiter=',', header=None)
# print(d2v200.head(10))

datasets_dict = {"Count Vectoriser": count_vec, "Doc2Vec50": d2v50, "Doc2Vec100": d2v100, "Doc2Vec200": d2v200}
y_train = train['rating']
# Meta without any of the non-numeric attributes
x_train = train.drop(['rating', 'date', 'review_id', 'reviewer_id', 'business_id'], axis=1)

# -------------- Logistic Regression for Doc2Vec200 and Doc2Vec100 with PCA(50) vs. Doc2Vec50
# Asking: Do higher dimensionality Doc2Vec text features perform better when reduced to the same dimensionality
# of smaller Doc2Vec features?
#
# avg_200 = []
# avg_100 = []
# avg_50 = []
#
# #Run across multiple random states
# for ran_state in range(10):
#
#     ### Doc2Vec200 with PCA = 50
#     kf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=ran_state)
#     pca = PCA(n_components=50, random_state=ran_state)
#     lgr = LogisticRegression()
#     #np.logspace(-4, 4, 4) for C
#     scores_200 = []
#     for train, test in kf.split(d2v200, y_train):
#         d2v200_reduced = pca.fit_transform(d2v200.loc[train])
#         lgr.fit(d2v200_reduced, y_train.loc[train])
#         test_reduced = pca.transform(d2v200.loc[test])
#         scores_200.append(lgr.score(test_reduced, y_train.loc[test]))
#     avg_200.append(np.average(scores_200))
#     # print(f"Average Accuracy for Logistic Regression with Cross Validation (10), PCA(50) for Doc2Vec200: {np.average(scores_200)}")
#
#     ### Doc2Vec100 with PCA = 50
#     scores_100 = []
#     for train, test in kf.split(d2v100, y_train):
#         d2v100_reduced = pca.fit_transform(d2v100.loc[train])
#         lgr.fit(d2v100_reduced, y_train.loc[train])
#         test_reduced = pca.transform(d2v100.loc[test])
#         scores_100.append(lgr.score(test_reduced, y_train.loc[test]))
#     avg_100.append(np.average(scores_100))
#     # print(f"Average Accuracy for Logistic Regression with Cross Validation (10), PCA(50) for Doc2Vec200: {np.average(scores_200)}")
#
#     ## Doc2Vec50 no PCA
#     scores_50 = []
#     for train, test in kf.split(d2v50, y_train):
#         lgr.fit(d2v50.loc[train], y_train.loc[train])
#         scores_50.append(lgr.score(d2v50.loc[test], y_train.loc[test]))
#     avg_50.append(np.average(scores_50))
#
# # print(f"Average Accuracy for Logistic Regression with Cross Validation (10), PCA(50) for Doc2Vec200: {np.average(avg_200)}")
# # print(f"Average Accuracy for Logistic Regression with Cross Validation (10), PCA(50) for Doc2Vec100: {np.average(avg_100)}")
# # print(f"Average Accuracy for Logistic Regression with Cross Validation (10) for Doc2Vec50: {np.average(avg_50)}")
#
# lgr_results = np.array((np.average(avg_200), np.average(avg_100), np.average(avg_50)))
# np.save('./results/lgr_for_pca50_vs_d2v50.npy', lgr_results)

# Can simply load to produce graph
lgr_results = np.load('./results/lgr_for_pca50_vs_d2v50.npy')
plt.rcParams['figure.figsize'] = [10, 7]
X = np.arange(3)
ind = X
# fig = plt.figure()
# ax = fig.add_axes([0, 0, 1, 1])
plt.ylim(0.8, 0.82)
bars = plt.bar(X + 0.00, lgr_results, color = 'royalblue', width=0.5)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(bars)


plt.ylabel('Prediction Accuracy', size=12)
plt.xlabel('Text Features', size=12)
plt.title('Logistic Regression Accuracy for High Dimension Doc2Vec \n reduced using PCA vs. Lower Dimension Doc2Vec', weight='bold',size=14)
plt.xticks(ind, ('Doc2Vec200 PCA(n=50)',
                 'Doc2Vec100 PCA(n=50)',
                 'Doc2Vec50'), size=10)
plt.show()
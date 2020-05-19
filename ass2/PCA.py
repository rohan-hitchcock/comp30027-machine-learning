import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
plt.rcParams['figure.figsize'] = [10, 7]
import numpy as np
import scipy
import pandas as pd

# Meta training data
train = pd.read_csv(r"./datasets/review_meta_train.csv")
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

# PCA
n = 2
pca = PCA(n_components=n)
tsvd = TruncatedSVD(n_components=n)
datasets_dict = {"Count Vectoriser": count_vec, "Doc2Vec50": d2v50, "Doc2Vec100": d2v100, "Doc2Vec200": d2v200}
for name, dataset in datasets_dict.items():

    if name == "Count Vectoriser":
        dataset_reduced = tsvd.fit_transform(dataset)
        print(f"Variance explained by each PC for {name}: ", tsvd.explained_variance_ratio_)
        print(f"Total variance: {np.sum(tsvd.explained_variance_ratio_)}")
    else:
        dataset_reduced = pca.fit_transform(dataset)
        print(f"Variance explained by each PC for {name}: ", pca.explained_variance_ratio_)
        print(f"Total variance: {np.sum(pca.explained_variance_ratio_)}")

    # "s" changes size of circles

    plt.scatter(dataset_reduced[low_rating, 0], dataset_reduced[low_rating, 1], c='red',
                s=1, label='Low Rating')
    plt.scatter(dataset_reduced[med_rating, 0], dataset_reduced[med_rating, 1], c='orange',
                s=1, label='Med Rating')
    plt.scatter(dataset_reduced[high_rating, 0], dataset_reduced[high_rating, 1], c='green',
                s=1, label='High Rating')
    plt.xlabel('1st Principal Component', size=12)
    plt.ylabel('2nd Principal Component', size=12)
    plt.legend(loc='upper left', shadow=True, title="Rating",
               title_fontsize=12)
    plt.title(f"PCA for {name} with {n} Components", weight="bold", size=14)
    plt.show()



import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

plt.rcParams['figure.figsize'] = [10, 7]
import numpy as np
import scipy
import pandas as pd

# ------- Plotting graphs for smaller number of principle components
# If changing n will have to change graphs
def plot_pca_components(datasets_dict, low_rating, med_rating, high_rating):
    n = 3

    pca = PCA(n_components=n)
    tsvd = TruncatedSVD(n_components=n)

    for name, dataset in datasets_dict.items():
        fig = plt.figure()
        #PCA cant be used for Sparse matrices, but seems truncatedSVD is better for text processing anyway? Need to read more
        if name == "Count Vectoriser":
            dataset_reduced = tsvd.fit_transform(dataset)
            print(f"Variance explained by each PC for {name}: ", tsvd.explained_variance_ratio_)
            print(f"Total variance: {np.sum(tsvd.explained_variance_ratio_)}")
        else:
            dataset_reduced = pca.fit_transform(dataset)
            print(f"Variance explained by each PC for {name}: ", pca.explained_variance_ratio_)
            print(f"Total variance: {np.sum(pca.explained_variance_ratio_)}")

        # "s" changes size of circles
        ax = fig.add_subplot(111, projection='3d')

        # reordering these would probably change which dots are in front of the rest
        ax.scatter(dataset_reduced[low_rating, 0], dataset_reduced[low_rating, 1], dataset_reduced[low_rating, 2], c='red',
                    s=1, label='Low Rating')
        ax.scatter(dataset_reduced[med_rating, 0], dataset_reduced[med_rating, 1], dataset_reduced[med_rating, 2], c='orange',
                    s=1, label='Med Rating')
        ax.scatter(dataset_reduced[high_rating, 0], dataset_reduced[high_rating, 1], dataset_reduced[high_rating, 2], c='green',
                    s=1, label='High Rating')
        ax.set_xlabel('1st Principal Component', size=12)
        ax.set_ylabel('2nd Principal Component', size=12)
        ax.set_zlabel('3rd Principal Component', size=12)
        ax.legend(loc='upper left', shadow=True, title="Rating",
                   title_fontsize=12)
        ax.set_title(f"PCA for {name} with {n} Components", weight="bold", size=14)
        plt.show()

# ----- Generating a graph for variance vs. n_components for each text feature
# Not to be used with CountVec
def plot_pca_components_vs_variance(datasets_dict, n_components, normalised=False):
    variances = {"Doc2Vec50": [], "Doc2Vec100": [], "Doc2Vec200": []}
    scaler = StandardScaler()
    for n in range(1, n_components+1):
        pca = PCA(n_components=n)
        for name, dataset in datasets_dict.items():
            if normalised:
                scaled = scaler.fit_transform(dataset)
                pca.fit(scaled)
                variances[name].append(np.sum(pca.explained_variance_ratio_))
            else:
                pca.fit(dataset)
                variances[name].append(np.sum(pca.explained_variance_ratio_))

    plt.plot(range(1, n_components+1), variances["Doc2Vec50"], 'bo', linestyle='-', label="Doc2Vec50")
    plt.plot(range(1, n_components+1), variances["Doc2Vec100"], 'go', linestyle='-', label="Doc2Vec100")
    plt.plot(range(1, n_components+1), variances["Doc2Vec200"], 'ko', linestyle='-', label="Doc2Vec200")
    plt.legend(loc='upper left', shadow=True, title="Text Feature", title_fontsize=12)
    plt.xlabel("Number of Components", size=12)
    plt.ylabel("Total Variance", size=12)
    if normalised:
        plt.title("Normalised PCA Variance vs. Number of Components", weight="bold", size=14)
    else:
        plt.title("Unnormalised PCA Variance vs. Number of Components", weight="bold", size=14)
    plt.show()


if __name__ == "__main__":

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

    # -------------- PCA
    datasets_dict = {"Count Vectoriser": count_vec, "Doc2Vec50": d2v50, "Doc2Vec100": d2v100, "Doc2Vec200": d2v200}
    datasets_dict_noCV = {"Doc2Vec50": d2v50, "Doc2Vec100": d2v100, "Doc2Vec200": d2v200}
    
    #plot_pca_components(datasets_dict, low_rating, med_rating, high_rating)
    #plot_pca_components_vs_variance(datasets_dict_noCV, 50, normalised=False)


## Tested the effects or normalising (feature/column wise) each doc2vec output before PCA.
## resulting graphs are saved, but normalising actually resulted in lower variance explained at each iteration
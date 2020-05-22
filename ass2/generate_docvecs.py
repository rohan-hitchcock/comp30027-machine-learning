""" Most of this code is adapted from the supplied features.ipynb file"""

import numpy as np
import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
import os

RANDOM_STATE = 7


def get_dot2vec_split(dim):
    """ For loading the precomputed train-test split of doc2vec vectors of the 
        specified dimension"""
    split_dir = f"d2v_strat{dim}"

    Xtrain = pd.read_csv(f"./datasets/computed/{split_dir}/Xtrain.csv", index_col=0)
    Xtest = pd.read_csv(f"./datasets/computed/{split_dir}/Xtest.csv", index_col=0)
    ytest = pd.read_csv(f"./datasets/computed/{split_dir}/ytest.csv", delimiter=',', header=None)
    ytrain = pd.read_csv(f"./datasets/computed/{split_dir}/ytrain.csv", delimiter=',', header=None)

    return Xtrain, Xtest, ytrain[1], ytest[1]


# function to preprocess and tokenize text
def tokenize_corpus(txt, tokens_only=False):
    for i, line in enumerate(txt):
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def compute_doc_embedding(train_corpus, test, dimension):

    # tokenize a training corpus
    corpus = list(tokenize_corpus(train_corpus))

    # train doc2vec on the training corpus
    model = gensim.models.doc2vec.Doc2Vec(vector_size=dimension, min_count=2, epochs=40)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # tokenize new documents
    doc_test = list(tokenize_corpus(test, tokens_only=True))

    # generate embeddings for the new documents
    doc_test_emb = np.zeros((len(doc_test),dimension))
    for i in range(len(doc_test)):
        doc_test_emb[i,:] = model.infer_vector(doc_test[i])

    # tokenize new documents
    doc_train = list(tokenize_corpus(train_corpus, tokens_only=True))

    # generate embeddings for the new documents
    doc_train_emb = np.zeros((len(doc_train),dimension))
    for i in range(len(doc_train)):
        doc_train_emb[i,:] = model.infer_vector(doc_train[i])

    return doc_test_emb, doc_train_emb

# MAIN
if __name__ == "__main__":
    res_train_review_table = pd.read_csv(r"./datasets/review_text_train.csv", index_col = False, delimiter = ',', header=0)
    data = res_train_review_table['review']

    data_meta = pd.read_csv(r"./datasets/review_meta_train.csv")
    class_label = data_meta['rating']

    for dim in range(25, 301, 25):
        print(f"running for dim {dim}")

        X_train, X_test, y_train, y_test= train_test_split(data, class_label, test_size=0.2, random_state=RANDOM_STATE*dim, stratify=class_label)

        test_embedding, train_embedding = compute_doc_embedding(X_train, X_test, dim)
        
        os.makedirs(f"./datasets/computed/d2v_strat{dim}/")

        pd.DataFrame(test_embedding).to_csv(f"./datasets/computed/d2v_strat{dim}/Xtest.csv")
        pd.DataFrame(train_embedding).to_csv(f"./datasets/computed/d2v_strat{dim}/Xtrain.csv")
        y_train

        y_train.to_csv(f"./datasets/computed/d2v_strat{dim}/ytrain.csv")
        y_test.to_csv(f"./datasets/computed/d2v_strat{dim}/ytest.csv")

        del test_embedding
        del train_embedding


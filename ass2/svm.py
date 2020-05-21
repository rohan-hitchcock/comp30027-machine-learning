from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
import numpy as np
import pandas as pd

data = pd.read_csv(
    r"./datasets/review_text_features_doc2vec50/review_text_train_doc2vec50.csv", 
    index_col=False, delimiter=',', header=None
)
data_meta = pd.read_csv(r"./datasets/review_meta_train.csv")
class_label = data_meta['rating']

#specify the model here
model = svm.SVC(C=1, kernel='sigmoid')

cross_val = KFold(n_splits=5)

fscore = []
accuracy = []
precision = []
recall = []

for train_i, test_i in cross_val.split(data):


    data_train, data_test = data.iloc[train_i], data.iloc[test_i]
    class_label_train, class_label_test = class_label.iloc[train_i], class_label.iloc[test_i]

    model.fit(data_train, class_label_train)

    #prediction
    predict_label = model.predict(data_test)


    #evaluation
    #fscore.append(metrics.f1_score(class_label_test, predict_label))
    accuracy.append(metrics.accuracy_score(class_label_test, predict_label))
    #precision.append(metrics.precision_score(class_label_test, predict_label))
    #recall.append(metrics.recall_score(class_label_test, predict_label))


print(f"Accuracy: {np.mean(accuracy)}\n"
      f"Precision: {np.mean(precision)}\n"
      f"Recall: {np.mean(recall)}\n"
      f"F1Score: {np.mean(fscore)}")


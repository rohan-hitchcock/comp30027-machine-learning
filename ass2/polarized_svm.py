from sklearn import svm
import numpy as np

class PolarizedSVM:
    """ """

    def __init__(self, threshold, middle_class, kernel='linear', C=1.0):
        """ This classifier is an SVM based classifier designed to classify data 
            into three ordinal classes
            Args:
                threshold: the probability threshold a class must exceed to
                be classified as such.
                middle_class: the middle of the ordinal classes
                kernel: the kernel of the base SVM classifier
                C: the regularisation parameter.
        """
        self.threshold = threshold
        self.middle_class = middle_class
        self.model = svm.SVC(C=C, kernel=kernel, probability=True)


    def fit(self, X, y):
        
        index = y != self.middle_class

        X = X[index]
        y = y[index]

        self.model.fit(X, y)


    def predict(self, X):

        probs = self.model.predict_proba(X)
        predictions = np.empty(len(probs))

        for i, class_probs in enumerate(probs):
            
            c0_prob, c1_prob = class_probs

            if c0_prob < self.threshold and c1_prob < self.threshold:
                predictions[i] = self.middle_class
            
            else:
                predictions[i] = self.model.classes_[0] if c0_prob > c1_prob else self.model.classes_[1]

        return predictions


if __name__ == "__main__":

    from generate_docvecs import get_dot2vec_split
    from sklearn import metrics

    threshold = 0.85
    middle_class = 3
    C = 0.001

    dim = 125


    print(threshold)
    model = PolarizedSVM(threshold, middle_class, C=C)

    Xtrain, Xtest, ytrain, ytest = get_dot2vec_split(dim)

    model.fit(Xtrain, ytrain)

    predictions = model.predict(Xtest)


    fscore = metrics.f1_score(ytest, predictions, average='weighted')
    accuracy = metrics.accuracy_score(ytest, predictions)
    precision = metrics.precision_score(ytest, predictions, average='weighted')
    recall = metrics.recall_score(ytest, predictions, average='weighted')


    cm = metrics.confusion_matrix(ytest, predictions, labels=[1, 3, 5], normalize='true')

    print(f"Fscore: {fscore}\n"
          f"Accuracy: {accuracy}\n"
          f"Precision: {precision}\n"
          f"Recall: {recall}")


    print(cm)

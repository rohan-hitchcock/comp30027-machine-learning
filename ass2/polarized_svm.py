from sklearn import svm
import numpy as np

class PolarizedSVM:


    def __init__(self, threshold, middle_class, kernel='linear', C=1.0):
        self.threshold = threshold
        self.middle_class = middle_class
        self.model = svm.SVC(C=C, kernel=kernel)


    def fit(self, X, y):
        
        index = y != self.middle_class

        X = X[index]
        y = y[index]

        self.model.fit(X, y, probability=True)


    def predict(self, X):
        

        probs = self.model.predict_proba(X)

        print(probs)
        print(self.model.classes_)
        
        predictions = np.empty(len(probs))

        for i, class_probs in enumerate(probs):
            
            c0_prob, c1_prob = class_probs

            if c0_prob < self.threshold and c1_prob < self.threshold:
                predictions[i] = self.middle_class
            
            else:
                predictions[i] = self.model.classes_[0] if c0_prob > c1_prob else self.model.classes_[1]

        return predictions
        
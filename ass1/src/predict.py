import numpy as np
import pandas as pd


def predict(data, model, class_vals, priors):
    ybar = []
    for instance in data:
        label = predict_class_label(instance, model, class_vals, priors)
        ybar.append(label)
    return ybar


def predict_class_label(instance, model, class_vals, priors):
    likelihoods = dict()
    for c in class_vals:
        likelihoods[c] = priors[c]
        for attr in range(len(instance)):
            likelihoods[c] += np.log(model[attr + 1][c][instance[attr]])
    return max(likelihoods, key=likelihoods.get)

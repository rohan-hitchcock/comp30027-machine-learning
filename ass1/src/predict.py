import numpy as np


def predict(test, model, cnfg, priors):
    """Classifies new test data according to the niave bayers model and priors provided.
        returns the ground truth labels as well as the predicted labels."""
    class_col = cnfg['class_col']
    class_vals = np.unique(test[class_col])
    no_class = test.drop(axis=1, labels=[class_col]).to_numpy()
    ybar = []
    for instance in no_class:
        label = predict_class_label(instance, model, class_vals, priors)
        ybar.append(label)
    return test[class_col], ybar


def predict_class_label(instance, model, class_vals, priors):
    likelihoods = dict()
    for c in class_vals:
        likelihoods[c] = priors[c]
        for attr in range(len(instance)):
            likelihoods[c] += np.log(model[attr + 1][c][instance[attr]])
    return max(likelihoods, key=likelihoods.get)

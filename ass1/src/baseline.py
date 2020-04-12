import numpy as np
import src.niave_bayes as nb
import random
import pandas as pd

def classify_zero_r(training_class_obs, num_to_classify):
    """ Classifies according to the most common class in the training data.

        Args:
            training_class_obs: the observations of class in the training data
            num_to_classify: the number of instances to classify

        Returns:
            A numpy array of length num_to_classify containing the most frequent
            value in training_class_obs
    """
    values, counts = np.unique(training_class_obs, return_counts=True)
    ind = np.argmax(counts)
    return np.full(num_to_classify, values[ind])

def classify_random(training_class_obs, num_to_classify):
    """ Classifies data at random according to the relative frequencies of 
        each class in the training observations.

        Args:
            training_class_obs: the observations of class in the training data
            num_to_classify: the number of instances to classify

        Returns:
            An array of length num_to_clasify of class values chosen at the 
            same probability as in training_class_obs
    """
    class_vals = np.unique(training_class_obs)
    class_probs = nb.discrete_probabilities(training_class_obs, class_vals)
    return np.random.choice(class_vals, size=num_to_classify, p=class_probs)

def classify_uniform(training_class_obs, num_to_classify):
    """ Classifies data uniformly at random.

        Args:
            training_class_obs: the observations of class in the training data
            num_to_classify: the number of instances to classify

        Returns:
            An array of length num_to_clasify of class values where each value
            was chosen with uniform probability.
    """
    class_vals = np.unique(training_class_obs)
    return np.random.choice(class_vals, size=num_to_classify)

def classify_one_r(training_df, class_name, testing_df):

    
    min_error_rate = float("inf")

    best_attr = None
    best_predictor = None

    for attr in training_df.columns.drop(class_name):

        attr_groups = training_df[[attr, class_name]].groupby(attr).groups

        attr_predictor = dict()
        for av, idx in attr_groups.values:
            
            #get most frequent class for av, choosing randomly if more than one
            most_frequent_class = training_df[class_name][idx].mode()
            choice = np.random.randint(len(most_frequent_class))

            attr_predictor[av] = most_frequent_class.iloc[choice]

        #check error rate of this attribute as predictor against training data
        attr_predictions = np.empty(len(training_df))
        for i, av in enumerate(training_df[attr]):
            attr_predictions[i] = attr_predictor[av]

        error_rate = np.count_nonzero(attr_predictions == training_df[class_name])

        if error_rate < min_error_rate:
            min_error_rate = error_rate
            best_attr = attr
            best_predictor = attr_predictor
    
    test_predictions = pd.Series(
        np.empty(len(testing_df), dtype=training_df[class_name].dtype), 
        index=testing_df.index)

    for i, row in testing_df.rows():
        test_predictions[i] = best_predictor[row[best_attr]]
    
    return test_predictions
    


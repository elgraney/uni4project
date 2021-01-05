from numpy import logspace
import glob
import time
import os
import sys
import itertools
import pickle
import collections
from sklearn.linear_model import LogisticRegression
import constants
import evaluation
import commonFunctions
from sklearn import preprocessing
import numpy as np
import machineLearning

def filter_data_by_procedure(procedure, data, test_index):
    '''
    inputs:
    procedure: an array of booleans representing the features and their inclusion
    data: the dictionary of data items organised by feature

    Data is organised by feature, but needs to be organised by item for training.
    This function reorganises data, removing excluded features and extracting the category of each item.
    '''
    new_data = []
    categories = []
    for index in range(len(data[constants.feature0])): #for every item of data
        data_item = []
        for feature in data: # for every key / feature
            if feature != constants.feature0: # if the key is not category
                if procedure[feature][test_index]: # if the test says to include this feature
                    data_item.append(data[feature][index]) # include this feature
                    
        categories.append(data[constants.feature0][index])
        new_data.append(data_item)
    return new_data, categories


def test_order(features):
    '''
    Produce a list of booleans to represent every possible combination of features
    '''
    feature_dict = {}
    for feature in features:
        if feature == "mean": # might need to reword if feature names change
            feature_dict[feature] = [True]
        else:
            feature_dict[feature] = [False]

    return feature_dict


def get_test_id():
    test_id = "Logistic Regression with mean"

    return test_id


def test_features(procedure, test_index):
    test = []
    for key in procedure.keys():
        if key!="category":
            test.append(procedure[key][test_index])
    return test


def trainRegressor(training_data, training_categories):
    model = LogisticRegression(random_state = 0)

    model.fit(training_data, training_categories)
    return model


def output_logs(logs, save_dir):
    log_string = "\n".join(str(log) for log in logs)
    commonFunctions.text_output(log_string, "Logs", save_dir)


def output_stats(test_results, save_dir):
    output_string = ""
    for key, value in test_results.items():
        output_string += "\n{}: {}".format(key, value)

    commonFunctions.text_output(output_string[1:], "Statistics", save_dir)



if __name__ == '__main__':
    start = time.time()

    save_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Outputs", "Unique Code")
    commonFunctions.makedir(save_dir)

    data_set_dir = "V:\\Uni4\\SoloProject\\DataSets\\4_3_500_5_3_10_C_False_500_0.001_10_10_25_3\\2" # temp hard-coded
    data = pickle.load( open( data_set_dir, "rb") )

    data = machineLearning.standard_scale(data)

    
    training_set, test_set = commonFunctions.split_data_set(data)

    features = list(data.keys())[1:]
    print("Estimating with features {}".format(features))
    # We just want category and average magnitude
    # Category, magnitude pairs

    procedure = test_order(features)

    
    results = {}

    if True: #lazy indent
        test_bools = test_features(procedure, 0) # only one test
        test_id = get_test_id()
        test_save_dir = os.path.join(save_dir, test_id)
        commonFunctions.makedir(test_save_dir)
        results[test_id] = {}

        test_output = []
        for repeat in range(constants.training_repetitions-4):
            

            training_data, training_categories = filter_data_by_procedure(procedure, training_set, 0)
            test_data, test_categories = filter_data_by_procedure(procedure, test_set, 0)

            model = trainRegressor(training_data, training_categories)

            test_output += evaluation.test(model, test_data, test_categories)

        #eval metrics
        results[test_id]["exact_accuracy"] = evaluation.exact_accuracy(test_output)
        results[test_id]["lenient_accuracy"] = evaluation.lenient_accuracy(test_output)
        results[test_id]["vector_differences"], results[test_id]["scalar_differences"] = evaluation.average_difference(test_output)
        #results[test_id]["differences_distribution"] = evaluation.plot_differences_distribution(test_output, test_save_dir)
        #results[test_id]["wind_force_differences_distribution"] = evaluation.plot_differences_by_wind_force(test_output, test_save_dir)

    output_logs(test_output, test_save_dir)
    output_stats(results[test_id], test_save_dir)


    
    #sorted_results = collections.OrderedDict(sorted(results.items(), key=lambda x: x[1]))
    #for key,value in sorted_results.items():
        #output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Total differences/total items={}".format(key, value[0], value[1], value[2])
        #text_output(output_string, "Best", save_dir, features)
        

    end=time.time()
    print("estimation duration:")
    print(str(end - start))

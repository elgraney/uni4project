from numpy import logspace
import glob
import time
import os
import sys
import itertools
import pickle
import collections
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import svm
import constants
import evaluation
import commonFunctions
import machineLearning
import numpy as np

# This file handles the creation of a classifier and the assessment of its performance and the performance of each of the training features
def test_order(features):
    '''
    Produce a list of booleans to represent every possible combination of features
    '''
    feature_dict = {}
    unique_features = []
    for feature in features:
        feature_dict[feature] = []
        if not "SD" in feature:
            unique_features.append(feature)

    features_number = len(unique_features)
    all_procedures = list(itertools.product([True, False], repeat=features_number)) # list every combination of True and False for a list as long as the features number

    for test in all_procedures:
        for feature, test_val in zip(unique_features, test):
            feature_dict[feature].append(test_val)
            feature_dict[feature+"SD"].append(test_val)

    return feature_dict


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



def test_features(procedure, test_index):
    test = []
    for key in procedure.keys():
        if key!="category":
            test.append(procedure[key][test_index])
    return test
    

def trainSVM(training_data, training_categories, kernel = 'rbf', gamma = 'auto', C=1):
    model = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    model.fit(training_data, training_categories)
    return model


def get_test_id(test, features):
    test_id = ""
    for boolean, feature in zip(test, features):
        if boolean:
            test_id = test_id +","+ feature

    test_id = test_id[2:]
    return test_id


def output_logs(logs, save_dir):
    log_string = "\n".join(str(log) for log in logs)
    commonFunctions.text_output(log_string, "Logs", save_dir)


def output_stats(test_results, save_dir):
    output_string = ""
    for key, value in test_results.items():
        output_string += "\n{}: {}".format(key, value)

    commonFunctions.text_output(output_string[1:], "Statistics", save_dir)


def input_svm_params(args):
    if len(args) > 4:
        try:
            svm_code = args[4].split("_")
            kernel = svm_code[0]
            try:
                gamma = float(svm_code[1])
            except:
                gamma = str(svm_code[1])
            C = float(svm_code[2])

        except:
            print("Error in input string: using default settings")
    else:
        kernel = "rbf"
        gamma = "auto"
        C = 1
        pass
    svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))

    return  svm_code, kernel, gamma, C

# TODO make neat and incorporate evaluation.py
# allow param input
# How to save a model?
# What do we want to test here? Feautre sets. Model params.
# We need to generate random dataset partitions based on forces
# We need more functions:
#   data set split (separate file)
#   set procedure (separate file)
#   train model
#   test accuracy
#   evaluation metrics


if __name__ == '__main__':
    start = time.time()

    preprocessing_code, opflow_code, filename = commonFunctions.code_inputs(sys.argv)

    svm_code, kernel, gamma, C = input_svm_params(sys.argv)

    save_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Outputs", preprocessing_code)
    commonFunctions.makedir(save_dir)
    save_dir = os.path.join(save_dir, opflow_code)
    commonFunctions.makedir(save_dir)

    if not os.path.exists(os.path.join(save_dir, svm_code)):
        commonFunctions.makedir(os.path.join(save_dir, svm_code))
    else:
        print("Save directory already exists")
        exit()

    load_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Datasets", preprocessing_code, opflow_code, filename)
    data = pickle.load( open( load_dir, "rb") )

    #Scale data
    data = machineLearning.normal_scale(data)
    data = machineLearning.standard_scale(data)

    
    training_set, test_set = commonFunctions.split_data_set(data)

    features = list(data.keys())[1:]
    print("Estimating with features {}".format(features))
    procedure = test_order(features)
    

    results = {}
    test_indices = range(len(list(procedure.values())[0])-1)
    if len(test_indices) > 1000:
        print("Limiting to 1000")
        test_indices = list(np.random.choice(test_indices, 1000))

    for test_index in test_indices: # Each test (max of 1000)
        test_bools = test_features(procedure, test_index)
        test_id = get_test_id(test_bools, features)
        test_save_dir = os.path.join(save_dir, svm_code, test_id)
        commonFunctions.makedir(test_save_dir)
        results[test_id] = {}

        test_output = []
        for repeat in range(constants.training_repetitions):
            

            training_data, training_categories = filter_data_by_procedure(procedure, training_set, test_index)
            test_data, test_categories = filter_data_by_procedure(procedure, test_set, test_index)

            model = trainSVM(training_data, training_categories)

            test_output += evaluation.test(model, test_data, test_categories)

        #eval metrics
        results[test_id]["exact_accuracy"] = evaluation.exact_accuracy(test_output)
        results[test_id]["lenient_accuracy"] = evaluation.lenient_accuracy(test_output)
        results[test_id]["vector_differences"], results[test_id]["scalar_differences"] = evaluation.average_difference(test_output)
        #results[test_id]["differences_distribution"] = evaluation.plot_differences_distribution(test_output, test_save_dir)
        #results[test_id]["wind_force_differences_distribution"] = evaluation.plot_differences_by_wind_force(test_output, test_save_dir)

        output_logs(test_output, test_save_dir)
        output_stats(results[test_id], test_save_dir)


        
    evaluation.test_ranking(os.path.join(save_dir, svm_code))

    end=time.time()
    print("estimation duration:")
    print(str(end - start))
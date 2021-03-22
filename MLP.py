from numpy import logspace
import glob
import time
import os
import sys
import itertools
import pickle
import collections
from sklearn.neural_network import MLPClassifier
import constants
import evaluation
import commonFunctions
import machineLearning as ML
import numpy as np

# This file handles the creation of a classifier and the assessment of its performance and the performance of each of the training features    

def trainMLP(training_data, training_categories):
    clf = MLPClassifier()
    clf.fit(training_data, training_categories)
    return clf


def input_ml_params(args):
    if len(args) > 4:
        try:
            ml_code = args[4].split("_")
            kernel = ml_code[0]
            try:
                gamma = float(ml_code[1])
            except:
                gamma = str(ml_code[1])
            C = float(ml_code[2])

        except:
            print("Error in input string: using default settings")
    else:
        kernel = "rbf"
        gamma = "auto"
        C = 1
        pass
    ml_code = "MLP_{}_{}_{}".format(str(kernel), str(gamma), str(C))

    return  ml_code, kernel, gamma, C

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
    ml_code, kernel, gamma, C = input_ml_params(sys.argv)
    data, save_dir = ML.setup_output(preprocessing_code, opflow_code,ml_code,filename)

    #Scale data - currently does normal then standard
    data = ML.normalisation(data)
    training_set, test_set = ML.split_data_set(data)

    features = list(data.keys())[1:]
    print("Estimating with features {}".format(features))
    procedure = ML.test_order(features)

    results = {}
    test_indices = range(len(list(procedure.values())[0])-1)
    if len(test_indices) > 1000:
        print("Limiting to 1000")
        test_indices = list(np.random.choice(test_indices, 1000))

    for test_index in test_indices: # Each test (max of 1000)
        test_bools = ML.test_features(procedure, test_index)
        test_id = ML.get_test_id(test_bools, features)
        test_save_dir = os.path.join(save_dir, ml_code, test_id)
        commonFunctions.makedir(test_save_dir)
        results[test_id] = {}

        test_output = []
        for repeat in range(constants.training_repetitions):
            

            training_data, training_categories = ML.filter_data_by_procedure(procedure, training_set, test_index)
            test_data, test_categories = ML.filter_data_by_procedure(procedure, test_set, test_index)

            model = trainMLP(training_data, training_categories)

            test_output += evaluation.test(model, test_data, test_categories)

        #eval metrics
        results[test_id]["exact_accuracy"] = evaluation.exact_accuracy(test_output)
        results[test_id]["lenient_accuracy"] = evaluation.lenient_accuracy(test_output)
        results[test_id]["vector_differences"], results[test_id]["scalar_differences"] = evaluation.average_difference(test_output)
        #results[test_id]["differences_distribution"] = evaluation.plot_differences_distribution(test_output, test_save_dir)
        #results[test_id]["wind_force_differences_distribution"] = evaluation.plot_differences_by_wind_force(test_output, test_save_dir)

        ML.output_logs(test_output, test_save_dir)
        ML.output_stats(results[test_id], test_save_dir)
        
    evaluation.test_ranking(os.path.join(save_dir, ml_code))

    end=time.time()
    print("estimation duration:")
    print(str(end - start))
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
import machineLearning as ML
import warnings
from matplotlib import pyplot as plt




def trainRegressor(training_data, training_categories):
    model = LogisticRegression()

    model.fit(training_data, training_categories)
    return model

def input_regression_params(args):
    
    ml_code = "Logistic_Regression"

    return  ml_code



if __name__ == '__main__':
    start = time.time()
    warnings.filterwarnings("ignore")

    preprocessing_code, opflow_code, filename = commonFunctions.code_inputs(sys.argv)
    ml_code = input_regression_params(sys.argv)
    data, save_dir = ML.setup_output(preprocessing_code, opflow_code, ml_code, filename)

    #Scale data
    data = ML.normalisation(data)
    training_set, test_set = ML.split_data_set(data, False)

    features = list(data.keys())[1:]
    print("Estimating with features {}".format(features))
    # We just want category and average magnitude
    # Category, magnitude pairs

    procedure = ML.test_order(features)

    
    results = {}
    models = {}
    test_indices = range(len(list(procedure.values())[0])-1)
    for test_index in test_indices: # Each test (max of 1000)
        test_bools = ML.test_features(procedure, test_index)
        test_id = ML.get_test_id(test_bools, features)
        test_save_dir = os.path.join(save_dir, ml_code, test_id)
        commonFunctions.makedir(test_save_dir)
        results[test_id] = {}
        models[test_id] = []


        test_output = []
        for repeat in range(constants.training_repetitions):
            

            training_data, training_categories = ML.filter_data_by_procedure(procedure, training_set, test_index)
            test_data, test_categories = ML.filter_data_by_procedure(procedure, test_set, test_index)


            model = trainRegressor(training_data, training_categories)
            test_output += evaluation.test(model, test_data, test_categories)
        
        models[test_id].append(model)
        #eval metrics
        results[test_id]["exact_accuracy"] = evaluation.exact_accuracy(test_output)
        results[test_id]["lenient_accuracy"] = evaluation.lenient_accuracy(test_output)
        results[test_id]["vector_differences"], results[test_id]["scalar_differences"] = evaluation.average_difference(test_output)
        results[test_id]["mean_squared_differences"] = evaluation.MS_difference(test_output)
        #results[test_id]["differences_distribution"] = evaluation.plot_differences_distribution(test_output, test_save_dir)
        #results[test_id]["wind_force_differences_distribution"] = evaluation.plot_differences_by_wind_force(test_output, test_save_dir)

        ML.output_logs(test_output, test_save_dir)
        ML.output_stats(results[test_id], test_save_dir)
    
    _, _, MSE_stats = evaluation.test_ranking(os.path.join(save_dir, ml_code))
    #evaluation.feature_importance_LR(models, MSE_stats)
    

    save_id = max(models.keys(), key=len)
    print(save_id)
    with open("Best_clf", 'wb') as out:
        pickle.dump(models[save_id], out)
    

    end=time.time()
    print("estimation duration:")
    print(str(end - start))

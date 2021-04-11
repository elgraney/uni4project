from sklearn import preprocessing
import constants
import numpy as np
import itertools
import commonFunctions
import os
import pickle
import random

def fix_NAN(value):
    for index in range(len(value)):
        try:
            eval(str(value[index]))
        except NameError:
            print(value[index])
            value[index] = 0

    return value

def standard_scale(data):
    for key, value in data.items():
        if key != constants.feature0:
            fix_NAN(value)
            scaler = preprocessing.StandardScaler().fit(np.array(value).reshape(-1,1))
            data[key] = list((scaler.transform(np.array(value).reshape(-1,1))).flatten())
    return data


def normal_scale(data):
    for key, value in data.items():
        if key != constants.feature0:
            fix_NAN(value)
            scaler = preprocessing.MinMaxScaler().fit(np.array(value).reshape(-1,1))
            data[key] = list((scaler.transform(np.array(value).reshape(-1,1))).flatten())
    return data


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


def setup_output(preprocessing_code, opflow_code, svm_code, filename):
    save_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Outputs", preprocessing_code)
    commonFunctions.makedir(save_dir)
    save_dir = os.path.join(save_dir, opflow_code)
    commonFunctions.makedir(save_dir)

    if not os.path.exists(os.path.join(save_dir, svm_code)):
        commonFunctions.makedir(os.path.join(save_dir, svm_code))
    else:
        print("Save directory already exists with code", svm_code)
        exit()

    load_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Datasets", preprocessing_code, opflow_code, filename)
    data = pickle.load( open( load_dir, "rb") )

    return data, save_dir


def normalisation(data):
    data = normal_scale(data)
    data = standard_scale(data)
    return data


def split_data_set(data, equal = True): 
    indices_by_force = [[] for x in range(13)] 
    for index in range(len(data[constants.feature0])):
        try:
            indices_by_force[int(data[constants.feature0][index])].append(index)
        except:
            print("bad id: " + data[constants.feature0][index])

    training_data = commonFunctions.clear_dict_items(data.copy())
    test_data = commonFunctions.clear_dict_items(data.copy())
    validation_data = commonFunctions.clear_dict_items(data.copy())
    if equal:
        smallest_catgory = min([len(lst) for lst in indices_by_force])
        training_per_force = round(smallest_catgory * constants.training_set_proportion)
        validation_partition = 0
        for lst in indices_by_force:
            random.shuffle(lst)
            for feature in data.keys():
                validation_data[feature] += [data[feature][index] for index in lst[:validation_partition]]
                training_data[feature] += [data[feature][index] for index in lst[validation_partition:training_per_force]]
                test_data[feature] += [data[feature][index] for index in lst[training_per_force:smallest_catgory]]

    else:
        for lst in indices_by_force:
            random.shuffle(lst)
            validation_partition = round(len(lst) * constants.validation_set_proportion)
            test_partition = round(len(lst) * constants.test_set_proportion)

            for feature in data.keys():
                validation_data[feature] += [data[feature][index] for index in lst[:validation_partition]]
                test_data[feature] += [data[feature][index] for index in lst[validation_partition:test_partition]]
                training_data[feature] += [data[feature][index] for index in lst[test_partition:]]
    
    return training_data, test_data #,validation_data
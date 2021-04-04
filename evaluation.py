import itertools
from matplotlib import pyplot as plt
import os
import commonFunctions
import numpy as np
import pylab
import statistics
import pickle

# How do we deal with the data?
# 1. Store training and test separately
# 2. Store dataset as one and use indexing to mark what is used for what. 

# Train model in separate file. Therefore save and import model.

# This is a class of functions for testing a model.

def test_order(features):
    '''
    Produce a list of booleans to represent every possible combination of features
    '''
    feature_dict = {}
    for feature in features:
        feature_dict[feature] = []

    features_number = len(features)
    all_procedures = list(itertools.product([True, False], repeat=features_number)) # list every combination of True and False for a list as long as the features number

    for test in all_procedures:
        for feature, test_val in zip(features, test):
            feature_dict[feature].append(test_val)
    return feature_dict


def MME_test_ranking(load_dir, output = True):
    # Feature search takes place in the model selection phase and the best 10 or so models are saved. 
    # So delete this
    exact_stats = {}
    lenient_stats = {}
    MME_stats = {}
    for test in os.listdir(load_dir):
        with open(os.path.join(load_dir, test, "Statistics.txt")) as stats_file:
            file_id = test
            exact_stats[file_id] = stats_file.readline().split(" ")[1]
            lenient_stats[file_id] = stats_file.readline().split(" ")[1]
            print(stats_file.readline())
            print(stats_file.readline())
            MME_stats[file_id] = stats_file.readline().split(" ")[1]

    exact_stats = dict(sorted(exact_stats.items(), key=lambda item: item[1]))
    lenient_stats = dict(sorted(lenient_stats.items(), key=lambda item: item[1]))
    MME_stats = dict(sorted(MME_stats.items(), key=lambda item: item[1]))
    
    if output:
        keys = list(MME_stats.keys())[-10:]
        for key in keys:
            value = str(round(float(MME_stats[key]), 3))
            acc = str(round(float(lenient_stats[key]), 3))
            print(str(key)+": "+value+", "+acc+"%")
    return exact_stats, lenient_stats
    # TODO NEEDS A WHOLE LOT OF WORK!


def test_ranking(load_dir, output = True):
    # Feature search takes place in the model selection phase and the best 10 or so models are saved. 
    # So delete this
    exact_stats = {}
    lenient_stats = {}
    for test in os.listdir(load_dir):
        with open(os.path.join(load_dir, test, "Statistics.txt")) as stats_file:
            file_id = test
            exact_stats[file_id] = stats_file.readline().split(" ")[1]
            lenient_stats[file_id] = stats_file.readline().split(" ")[1]

    exact_stats = dict(sorted(exact_stats.items(), key=lambda item: item[1]))
    lenient_stats = dict(sorted(lenient_stats.items(), key=lambda item: item[1]))
    
    if output:
        keys = list(lenient_stats.keys())[-10:]
        values = list(lenient_stats.values())[-10:]
        for key, value in zip(keys, values):
            value = str(round(float(value), 3))
            print(str(key)+": "+value+"%")
    return exact_stats, lenient_stats


def test(model, test_data, test_categories):
    # send in test data and apply to model
    # This is where we fully evaluate the model
    # All different stats must be made here

    return_set = []
    for index in range(len(test_data)):
        item = {}
        estimate = model.predict([test_data[index]])
        actual = test_categories[index]

        item["estimate"] = eval(estimate[0])
        item["actual"] = eval(actual)
        item["Meta placeholder"] = "Placeholder"

        return_set.append(item)
    return return_set


def exact_accuracy(test_output):
    total = 0
    for item in test_output:
        if item["estimate"] == item["actual"]:
            total += 1
    
    return total / len(test_output) *100


def lenient_accuracy(test_output):
    total = 0
    for item in test_output:
        difference = abs(item["estimate"]-item["actual"])
        if difference<=1:
            total +=1
    
    return total / len(test_output) * 100


def average_difference(test_output):
    scalar_diff = 0
    vector_diff = 0
    for item in test_output:
        diff = item["estimate"]-item["actual"]
        vector_diff += diff
        scalar_diff += abs(diff)
    
    return vector_diff / len(test_output), scalar_diff / len(test_output)


def MS_difference(test_output):
    diff = 0
    for item in test_output:
        diff += (item["estimate"]-item["actual"])**2

    return diff / len(test_output)


def plot_differences_distribution(test_output, save_dir):
    plt.clf()

    differences = [0 for x in range(13)]
    for item in test_output:
        difference = abs(item["estimate"]-item["actual"])
        differences[difference] += 1
    
    plt.plot(differences)
    plt.savefig(os.path.join(save_dir, "Difference Distribution"))
    return differences
 

def plot_differences_by_wind_force(test_output, save_dir):
    plt.clf()

    forces = [[] for x in range(13)]
    for item in test_output:
        difference = abs(item["estimate"]-item["actual"])
        forces[item["actual"]].append(difference)

    plot = [0 for x in range(13)]
    for index in range(len(forces)):
        plot[index] = sum(forces[index])/len(forces[index])
    
    plt.plot(plot)
    plt.savefig(os.path.join(save_dir, "Average Differences organised by Force"))
    return plot

def feature_average_by_category(path):
    data = pickle.load( open( path, "rb") )

    stats_by_force = {}

    for feature in data.keys():
        if feature != "category":
            stats_by_force[feature] = [[] for x in range(13)] # indices are the force

    for index in range(len(data[list(data.keys())[0]])):
 
        force = data["category"][index]
        for feature in data.keys():
            if feature != "category":
                try:
                    stats_by_force[feature][eval(force)].append(data[feature][index])
                except Exception:
                    continue

    for feature in stats_by_force.keys():
        for force in range(13):
            if stats_by_force[feature][force] != []:
                stats_by_force[feature][force] = statistics.mean(stats_by_force[feature][force])
            else: stats_by_force[feature][force] = 0

    for stat in stats_by_force.keys():
        print(stat)
        plt.plot(stats_by_force[stat])
        plt.show()

 

if __name__ == "__main__":
    directory = "V:\\Uni4\\SoloProject\\Outputs\\"

    test_code_bests = {}
    best = {} # key = param index, value = (value, accuracy)

    for prepep_test in os.listdir(directory):
        if "v" not in prepep_test and prepep_test != "Unique_Code":
            
            for opflow_test in os.listdir(os.path.join(directory, prepep_test)):
                params = prepep_test.split("\\")[-1].split("_") + opflow_test.split("_")
                for SVM_test in os.listdir(os.path.join(directory, prepep_test,  opflow_test)):
                    params = prepep_test.split("\\")[-1].split("_") + opflow_test.split("_") + SVM_test.split("_")
                    #print(params)
                    exact_stats, lenient_stats = test_ranking(os.path.join(directory, prepep_test,  opflow_test, SVM_test), False)
                    try:
                        last = list(lenient_stats.keys())[-1]
                    except:
                        print(os.path.join(directory, prepep_test,  opflow_test, SVM_test))
                        exit()
                    #print(last, lenient_stats[last])

                    test_code_bests[(prepep_test+"__"+opflow_test+"__"+SVM_test)] = lenient_stats[last]
                    
                    for param_index in range(len(params)):
                        try:
                            best[param_index].append((params[param_index], lenient_stats[last]))
                        except:
                            best[param_index] = [(params[param_index], lenient_stats[last])]
    
    test_code_bests =  {k: v for k, v in sorted(test_code_bests.items(), key=lambda item: item[1])}
    

    for key, value in best.items():
        sorted_value = sorted(value, key=lambda val: val[1])  
        
        x = [item[0] for item in sorted_value]
        if len(set(x))> 1:
            y = [round(eval(item[1]),3) for item in sorted_value]

            plt.scatter(x, y)
            print(x, y)


            plt.title(("Param", key))
            plt.show()
                
    print(test_code_bests)        
            

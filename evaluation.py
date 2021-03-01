import itertools
from matplotlib import pyplot as plt
import os
import commonFunctions

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
        for key, value in lenient_stats.items():
            print(key, value)
    return exact_stats, lenient_stats
    # TODO NEEDS A WHOLE LOT OF WORK!


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
 

if __name__ == "__main__":
    directory = "V:\\Uni4\\SoloProject\\Outputs\\"
    for test_file in os.listdir("V:\\Uni4\\SoloProject\\Outputs\\"):
        if "v" not in test_file and test_file != "Unique_Code":
            print(test_file)
            exact_stats, lenient_stats = test_ranking(os.path.join(directory, test_file, "tests"), False)
            last = list(lenient_stats.keys())[-1]
            print(last, lenient_stats[last])

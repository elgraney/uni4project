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
import multiprocessing

# This file handles the creation of a classifier and the assessment of its performance and the performance of each of the training features

def text_output(text, name, path, features):
    '''
    Write a string to a text file
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path,name+'.txt')): # output the features used to the first line for reference
        with open(os.path.join(path,name+'.txt'),'a') as result_file:
            result_file.write(str(features))
    with open(os.path.join(path,name+'.txt'),'a') as result_file:
        result_file.write(text)
    result_file.close()

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


def trainSVC(test, training_data, training_categories, test_data, test_categories, differences, save_dir, features, folder):
    #performance variables
    exact_accuracy = 0
    lenient_accuracy = 0
    #Note: differences might include differences from other sets
    
    SVC = svm.SVC(kernel='rbf', gamma="auto", C=1)
    SVC.fit(training_data, training_categories) # train with training data
    
    for index in range(len(test_data)): # estimate category for each item of test data
        estimate = SVC.predict([test_data[index]])
        actual = test_categories[index]

        # record performance
        output_string = "\nEstimated: {}, Actual: {}".format(estimate[0], actual) #estimate[0]? Can we do them all at once?
        text_output(output_string, "{}_estimations".format(folder), save_dir, features)
        difference = abs(int(estimate[0])-int(actual))
        if difference<=1:
            lenient_accuracy +=1
            if difference == 0:
                exact_accuracy +=1
        differences.append(difference)

    #For avg diffs, only include differences added in last test; the last len(test_data) items
    avg_differences = sum(differences[-len(test_data):])/len(test_data)

    perc_exact_accuracy = exact_accuracy/len(test_data) *100
    perc_lenient_accuracy = lenient_accuracy/len(test_data) *100

    output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Average Difference={}".format(test, perc_exact_accuracy, perc_lenient_accuracy, avg_differences)
    
    text_output(output_string, "{}_Logs".format(folder), save_dir, features)

    return exact_accuracy, lenient_accuracy



def evalTest(procedure, test_index, return_dict, data_set_dir, save_dir, features):
    #performance variable for combined data sets
    differences = []
    exact_acc = 0
    len_acc = 0
   
    # Remove "category" from features in procedure to be considered for current test
    test = []
    for key in procedure.keys():
        if key!="category":
            test.append(procedure[key][test_index])
    test_string = ", ".join([str(item) for item in test])

    os.mkdir(os.path.join(save_dir, str(test)))
    save_dir = os.path.join(save_dir, str(test))

    for folder in os.listdir(data_set_dir): # for each training-test set pair available
        for data_file in glob.glob(os.path.join(data_set_dir,folder,"TestSet*")):
            with open(data_file, 'rb') as fp:
                test_set = pickle.load(fp)
        for data_file in glob.glob(os.path.join(data_set_dir,folder,"TrainingSet*")):
            with open(data_file, 'rb') as fp:
                training_set = pickle.load(fp)


        # Following aquires data for features included in this test
        training_data = []
        training_categories = []
        for index in range(len(training_set["category"])): #for every item of data
            training_data_item = []
            for feature in training_set: # for every key / feature
                if feature != "category" and feature in procedure.keys(): # if the key is not category
                    if procedure[feature][test_index]: # if the test says to include this feature
                        training_data_item.append(training_set[feature][index]) # include this feature
                    
            training_categories.append(training_set["category"][index])
            training_data.append(training_data_item)
            
        # Repeat for test set:
        test_data = []
        test_categories = []
        for index in range(len(test_set["category"])):
            test_data_item = []
            for feature in test_set:
                if feature != "category" and feature in procedure.keys():
                    if procedure[feature][test_index]:
                        test_data_item.append(test_set[feature][index])
            test_categories.append(test_set["category"][index])
            test_data.append(test_data_item)

        #Being testing
        try: 
            e, l = trainSVC(test, training_data, training_categories, test_data, test_categories, differences, save_dir, features, folder)
            exact_acc += e
            len_acc += l
            #TODO Calc differences form Differences list

        except Exception as err:
            print("failed with {}".format(err))

    

    avg_perc_exact_acc = exact_acc / len(differences) * 100
    avg_perc_len_acc = len_acc / len(differences) * 100
    avg_diffs = sum(differences) / len(differences)

    output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Average Difference={}".format(test, avg_perc_exact_acc, avg_perc_len_acc, avg_diffs)
    print(output_string[1:])
        
    #text_output(output_string, "Accuracy", save_dir, features)

    return_dict[test_index] = [avg_perc_exact_acc, avg_perc_len_acc, avg_diffs, test_string]


if __name__ == '__main__':
    start = time.time()

    save_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Outputs")

    # handle command line args
    flowtype = "DenseFlow"
    if len(sys.argv)>1:
        if sys.argv[1].lower() == "dense":
            flowtype = "DenseFlow"
        elif sys.argv[1].lower() == "points":
            flowtype = "PointsFlow"
        elif sys.argv[1].lower() == "tracks":
            flowtype = "TracksFlow"
        else:
            print(sys.argv[1])
            raise Exception("Bad argument; argument 1")
        if len(sys.argv)>2:
            try:
                save_dir = os.path.join(save_dir, sys.argv[2])
                if os.path.exists(save_dir):
                    print("folder", save_dir, "already exists")
                    folder_number = 1
                    new_save_dir = save_dir+str(folder_number)
                    while(os.path.exists(new_save_dir)):
                        print("folder", save_dir, "already exists")
                        folder_number+=1
                        new_save_dir = save_dir +str(folder_number)
                    save_dir = new_save_dir
                print("Saving in folder", save_dir)
                os.mkdir(save_dir)
            except IOError:
                print(sys.argv[2])
                raise Exception("Bad argument; argument 2")


    data_set_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], os.path.join("DataSets", flowtype))
    total_folders = len(os.listdir(data_set_dir)) * 2
    count = 0
    
    # Hardcode access to the first file to extract features
    default_path = "0\\TestSet0"
    features = list(pickle.load( open( os.path.join(data_set_dir, default_path), "rb" )).keys())
    print("Estimating with features {}".format(features))
    procedure = test_order(features[1:])
    results = {}

    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()    

    stats = {}
    for test_index in range(len(list(procedure.values())[0])-1): # for each test in list
        while len(threads) >= 10:
            for thread in threads:
                if not thread.is_alive():   
                    threads.remove(thread)
                    # Return and save each feature from the video using a thread-safe dictionary 
                    for key,value in return_dict.items():
                        if not key in stats.keys():
                            stats[key] = []
                        stats[key].append(value) 
                        del return_dict[key]

        p1 = multiprocessing.Process(target=evalTest, args=(procedure, test_index, return_dict, data_set_dir, save_dir, features))
        threads.append(p1)
        p1.start()
        
    for key, value in stats.items():
        results[value[3]] = value[:3] 
    
    sorted_results = collections.OrderedDict(sorted(results.items(), key=lambda x: x[1]))
    for key,value in sorted_results.items():
        output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Total differences/total items={}".format(key, value[0], value[1], value[2])
        text_output(output_string, "Best", save_dir, features)
        

    end=time.time()
    print("estimation duration:")
    print(str(end - start))
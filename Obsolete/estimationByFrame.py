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

# This file is a variation on 'estimation.py'
# It attempts to train a classifer on features caluated from each frame rather than each clip
# It did not perform well.

def text_output(text, name, path, features):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path,name+'.txt')):
        with open(os.path.join(path,name+'.txt'),'a') as result_file:
            result_file.write(str(features))
    with open(os.path.join(path,name+'.txt'),'a') as result_file:
        result_file.write(text)
    result_file.close()

def test_order(features):
    '''
    '''
    feature_dict = {}
    for feature in features:
        feature_dict[feature] = []

    features_number = len(features)
    all_procedures = list(itertools.product([True, False], repeat=features_number))

    for test in all_procedures:
        for feature, test_val in zip(features, test):
            feature_dict[feature].append(test_val)
    return feature_dict


if __name__ == '__main__':
    start = time.time()

    save_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Outputs")

    flowtype = "DenseFlow"
    if len(sys.argv)>1:
        if sys.argv[1].lower() == "dense":
            flowtype = "DenseFlow"
        elif sys.argv[1].lower() == "points":
            flowtype = "PointsFlow"
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
    
    #Hardcode access to the fist file to extract features
    default_path = "0\\TestSet0"
    features = list(pickle.load( open( os.path.join(data_set_dir, default_path), "rb" )).keys())
    print("Estimating with features {}".format(features))
    procedure = test_order(features[1:])

    test_set = {}
    training_set = {}

    results = {}
    for test_index in range(len(list(procedure.values())[0])-1): # for each test in list
        rbfs = []
        average_exact_accuracy =0
        average_lenient_accuracy = 0
        average_differences = 0
        for folder in os.listdir(data_set_dir):
            try:
                for data_file in glob.glob(os.path.join(data_set_dir,folder,"TestSet*")):
                    with open(data_file, 'rb') as fp:
                        test_set = pickle.load(fp)
                for data_file in glob.glob(os.path.join(data_set_dir,folder,"TrainingSet*")):
                    with open(data_file, 'rb') as fp:
                        training_set = pickle.load(fp)

                test = []
                for key in procedure.keys():
                    if key!="category":
                        test.append(procedure[key][test_index])

                training_data = []
                training_categories = []
                for index in range(len(training_set["category"])): #for every item of data
                    training_data_item = []
                    for feature in training_set: # for ever key / feature
                        if feature != "category" and feature in procedure.keys(): # if the key is not category
                            if procedure[feature][test_index]: # if the test says to add this feature
                                training_data_item.append(training_set[feature][index]) #add this feature
                        
                    training_categories.append(training_set["category"][index])
                    training_data.append(training_data_item)

                training_2 = []
                training_categories_2 = []
                for item, category in zip(training_data, training_categories):
                    max_length = min(len(lst) for lst in item)
                    for index in range(max_length):
                        new_item = []
                        for feature in item:
                            new_item.append(feature[index])
                        training_2.append(new_item)
                        training_categories_2.append(category)

                training_categories = training_categories_2
                training_data = training_2


                
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

                test_2 = []
                test_categories_2 = []
                for item, category in zip(test_data, test_categories):
                    max_length = min(len(lst) for lst in item)
                    for index in range(max_length):
                        new_item = []
                        for feature in item:
                            new_item.append(feature[index])
                        test_2.append(new_item)
                        test_categories_2.append(category)

                test_categories = test_categories_2
                test_data = test_2


                #Being testing
                #print("beginning param search")
                #print("Training data length {}, item length {}, categories length {}".format(len(training_data),len(training_data[0]), len(training_categories)))

                rbfs.append(svm.SVC(kernel='rbf', gamma="auto"))
                rbfs[-1].fit(training_data, training_categories)
                exact_accuracy = 0
                lenient_accuracy = 0
                differences =[]
                for index in range(len(test_data)):
                    estimate = rbfs[-1].predict([test_data[index]])
                    actual = test_categories[index]
                    output_string = "estimated: {}, actual: {}\n".format(estimate[0], actual)
                    text_output(output_string, "estimationResults", save_dir, features)
                    difference = abs(int(estimate[0])-int(actual))
                    if difference<=1:
                        lenient_accuracy +=1
                        if difference == 0:
                            exact_accuracy +=1
                    differences.append(difference)

                total_differences = sum(differences)/len(test_data)


                exact_accuracy = exact_accuracy/len(test_data) *100
                lenient_accuracy = lenient_accuracy/len(test_data) *100

                output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Total differences/total items={}".format(test, exact_accuracy, lenient_accuracy, total_differences)
                text_output(output_string, "Logs", save_dir, features)

                average_exact_accuracy += exact_accuracy
                average_lenient_accuracy += lenient_accuracy
                average_differences += total_differences
            except Exception as err:
                print("failed with {}".format(err))


        average_exact_accuracy = average_exact_accuracy / len(os.listdir(data_set_dir))
        average_lenient_accuracy = average_lenient_accuracy / len(os.listdir(data_set_dir))
        average_differences = average_differences / len(os.listdir(data_set_dir))  
        output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Total differences/total items={}".format(test, average_exact_accuracy, average_lenient_accuracy, average_differences)
        print(output_string[2:])
        text_output(output_string, "Accuracy", save_dir, features)
        
        string_test = ", ".join([str(item) for item in test])
        results[string_test] = [average_exact_accuracy, average_lenient_accuracy, average_differences]
    
    sorted_results = collections.OrderedDict(sorted(results.items(), key=lambda x: x[1]))
    for key,value in sorted_results.items():
        output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Total differences/total items={}".format(key, value[0], value[1], value[2])
        text_output(output_string, "Best", save_dir, features)
        

    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))
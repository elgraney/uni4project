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


def text_output(text, name, path, features):
    '''
    Write a string to a text file
    '''
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
    Produce a list of booleans to represent every possible combination of features
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

    # handle command line args
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
    for test_index in range(len(list(procedure.values())[0])-1): #Cfor each test in list
        x_rbfs = []
        y_rbfs = []
        average_exact_accuracy =0
        average_lenient_accuracy = 0
        average_differences = 0
        for folder in os.listdir(data_set_dir):
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

            x_training_data = []
            y_training_data = []
            training_categories = []
            for index in range(len(training_set["category"])): #for every item of data
                x_training_data_item = []
                y_training_data_item = []
                for feature in training_set: # for ever key / feature
                    if feature != "category" and feature in procedure.keys(): # if the key is not category
                        if procedure[feature][test_index]: # if the test says to add this feature
                            x_training_data_item.append(training_set[feature][index][0]) #add this feature
                            y_training_data_item.append(training_set[feature][index][1])
                training_categories.append(training_set["category"][index])
                x_training_data.append(x_training_data_item)
                y_training_data.append(y_training_data_item)
                
            x_test_data = []
            y_test_data = []
            test_categories = []
            for index in range(len(test_set["category"])):
                x_test_data_item = []
                y_test_data_item = []
                for feature in test_set:
                    if feature != "category" and feature in procedure.keys():
                        if procedure[feature][test_index]:
                            x_test_data_item.append(test_set[feature][index][0])
                            y_test_data_item.append(test_set[feature][index][1])
                test_categories.append(test_set["category"][index])
                x_test_data.append(x_test_data_item)
                y_test_data.append(y_test_data_item)

            #Being testing
            #print("beginning param search")
            #print("Training data length {}, item length {}, categories length {}".format(len(training_data),len(training_data[0]), len(training_categories)))

            
            
            try: 
                x_rbfs.append(svm.SVC(kernel='rbf', gamma="auto"))
                y_rbfs.append(svm.SVC(kernel='rbf', gamma="auto"))
                x_rbfs[-1].fit(x_training_data, training_categories)
                y_rbfs[-1].fit(y_training_data, training_categories)

                exact_accuracy = 0
                lenient_accuracy = 0
                differences =[]
                for index in range(len(x_test_data)):
                    x_estimate = x_rbfs[-1].predict([x_test_data[index]])
                    y_estimate = y_rbfs[-1].predict([y_test_data[index]])

                    estimate = (int(x_estimate[0]) + int(y_estimate[0])) / 2 
                    actual = test_categories[index]
                    output_string = "estimated: {}, actual: {}\n".format(estimate, actual)
                    text_output(output_string, "estimationResults", save_dir, features)
                    difference = abs(int(estimate)-int(actual))
                    if difference<=1:
                        lenient_accuracy +=1
                        if difference == 0:
                            exact_accuracy +=1
                    differences.append(difference)

                total_differences = sum(differences)/len(x_test_data)


                exact_accuracy = exact_accuracy/len(x_test_data) *100
                lenient_accuracy = lenient_accuracy/len(x_test_data) *100

                output_string = "\nTest {}: Exact Accuracy={}, Lenient Accuracy={}, Total differences/total items={}".format(test, exact_accuracy, lenient_accuracy, total_differences)
                text_output(output_string, "Logs", save_dir, features)

                average_exact_accuracy += exact_accuracy
                average_lenient_accuracy += lenient_accuracy
                average_differences += total_differences
            except Exception as err:
                print("failed with {}".format(err))
                exit()


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
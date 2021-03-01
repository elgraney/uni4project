import os
import constants
import random
import math


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def text_output(text, name, path):
    '''
    Write a string to a text file
    '''
    makedir(path)
    with open(os.path.join(path, name+'.txt'),'a') as result_file:
        result_file.write(text)
    result_file.close()


def clear_dict_items(dictionary):
    for key in dictionary.keys():
        dictionary[key] = []
    return dictionary 


def split_data_set(data): 
    
    indices_by_force = [[] for x in range(13)] 
    for index in range(len(data[constants.feature0])):
        try:
            indices_by_force[int(data[constants.feature0][index])].append(index)
        except:
            print("bad id: " + data[constants.feature0][index])

    training_data = clear_dict_items(data.copy())
    test_data = clear_dict_items(data.copy())
    validation_data = clear_dict_items(data.copy())
    for lst in indices_by_force:
        random.shuffle(lst)
        validation_partition = round(len(lst) * constants.validation_set_proportion)
        test_partition = round(len(lst) * constants.test_set_proportion)

        for feature in data.keys():
            validation_data[feature] += [data[feature][index] for index in lst[:validation_partition]]
            test_data[feature] += [data[feature][index] for index in lst[validation_partition:test_partition]]
            training_data[feature] += [data[feature][index] for index in lst[test_partition:]]
    
    return training_data, test_data #,validation_data


def code_inputs(sys_args):
    if len(sys_args) > 1:
        try:
            preprocessing_code = sys_args[1]
            opflow_code = str(sys_args[2])
            filename = str(sys_args[3])

        except:
            print("Error in input string: using default settings")
    else:
        preprocessing_code = "4_3_500_5_3_10_C_10_False"
        opflow_code = "500_0.001_10_10_25_3"
        filename = "default"

    return preprocessing_code, opflow_code, filename
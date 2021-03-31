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


def code_inputs(sys_args):
    if len(sys_args) > 1:
        try:
            preprocessing_code = sys_args[1]
            opflow_code = str(sys_args[2])
            filename = str(sys_args[3])

        except:
            print("Error in input string: using default settings")
    else:
        #preprocessing_code = "4_3_500_5_3_10_C_10"
        preprocessing_code = "4_3_300_1_3_1_C_1"
        #opflow_code = "500_0.001_10_10_25_3"
        opflow_code = "500_0.01_5_10_25_3"
        filename = "default"

    return preprocessing_code, opflow_code, filename

def mph_to_beaufort(mph):
    if mph < 1:
        return 0
    elif mph < 4:
        return 1
    elif mph < 8:
        return 2
    elif mph < 13:
        return 3
    elif mph < 19:
        return 4
    elif mph < 25:
        return 5
    elif mph < 32:
        return 6
    elif mph < 39:
        return 7
    elif mph < 47:
        return 8
    elif mph < 55:
        return 9
    elif mph < 64:
        return 10
    elif mph < 73:
        return 11
    else:
        return 12
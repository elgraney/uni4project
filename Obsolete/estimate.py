import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import time
import multiprocessing 
import random
import itertools
from sklearn import svm

def text_output(text):
    with open('V:\\Uni 3\\Project\\accuracy.txt','a') as result_file:
        result_file.write(text)
    result_file.close()


def mean_feature(index, test, data):
    if test[index]:
        return statistics.mean(data)
    else:
        return None

def max_feature(index, test, data):
    if test[index]:
        return max(data)
    else:
        return None
    

def processVideo(file, load_folder, test, return_dict):
    flow_list = np.load(os.path.join(load_folder, file))

    # Instantiate all variables, but some might not be used depending on the test.
    mean_list = [] # For index 0, 5, 6, 7, 8, 9, 10, 11
    median_grouped_list = [] # For index 1
    sd_list = [] # For index 2, 12
    max_value = [] # For index 3, 4, 7, 8
    difference_from_max = [] # index 3, 4
    difference_from_mean =[] # index 5, 6
    relative_difference_from_mean = [] # for index 9, 10
    percentage_difference = [] # For index 11
    relative_difference_from_max = [] # For index 7, 8
    sd_change = [] # for index 12

    name = file.split(".")[0]

    print("processing frames from video", file)
    for frame_index in range(len(flow_list)):
        #get optical flow from 2 frames and save changes in magnitudes
        magnitudes = [0.0]  
        #Debugging step 2  CURRENTLY IN
        for x in range(0, len((flow_list[0][0]))):
            for y in range(0, len((flow_list[0]))):
                # Optional ignore low values || REQUIRED TO AVOID DIVIDE BY ZERO
                magnitude = float(np.linalg.norm(flow_list[frame_index][y][x]))
                if magnitude > 0.001:
                    magnitudes.append(magnitude)

        if test[0] or test[5] or test[6] or test[7] or test[8] or test[9] or test[10] or test[11]:
            mean_list.append(statistics.mean(magnitudes)) # mean of whole flow
            if len(mean_list)>1:
                difference_from_mean.append(abs(mean_list[-1] - mean_list[-2])) # Difference between this frames mean val and last frame's mean val

        if test[1]: 
            median_grouped_list.append(statistics.median_grouped(magnitudes)) #median of all flow
        
        if test[2] or test[12]:
            sd_list.append(statistics.pstdev(magnitudes)) # sd of all flow

        if test[3] or test[4] or test[7] or test[8]:
            max_value.append(max(magnitudes)) # highest flow val
            if len(max_value)>1:
                difference_from_max.append(abs(max_value[-1] - max_value[-2])) # Difference between this frames max val and last frame's max val

        '''
        # Only process up to certain number of frames (for debugging)
        # Debugging
        if frame_index > 8:
            print("debugging; first 15 frames only.")
            break
        '''
        
        

    if test[7] or test[8]:
        # For each max frame difference calculate the distance relative to mean of whole video
        for item in difference_from_max:
            relative_difference_from_max.append(item / statistics.mean(mean_list))
    
    if test[9] or test[10]:
        #for each mean frame difference calculate the distance relative to mean of whole video
        for item in difference_from_mean:
            relative_difference_from_mean.append(item / statistics.mean(mean_list))

    if test[11]:
        for item_index in range(2, len(mean_list)):
            if sd_list[item_index-1] != 0:
                percentage_difference.append((mean_list[item_index]/mean_list[item_index-1])*100) 
            else:   
                percentage_difference.append(0.0) 

    if test[12]:
            # For every item in the standard deviation list, find the change and save it.       
            for item_index in range(2, len(sd_list)):
                sd_change.append(abs(sd_list[item_index-1]-sd_list[item_index]))
            
    video_windspeed = name.split("-")[-1]

    features_list =[int(video_windspeed)]
    if test[0]:
        features_list.append(mean_feature(0, test, mean_list))
    if test[1]:    
        features_list.append(mean_feature(1, test, median_grouped_list))
    if test[2]:    
        features_list.append(mean_feature(2, test, sd_list))
    if test[3]:    
        features_list.append(mean_feature(3, test, difference_from_max))
    if test[4]:    
        features_list.append(max_feature(4, test, difference_from_max))
    if test[5]:    
        features_list.append(mean_feature(5, test, difference_from_mean))
    if test[6]:    
        features_list.append(max_feature(6, test, difference_from_mean))
    if test[7]:    
        features_list.append(mean_feature(7, test, relative_difference_from_max))
    if test[8]:    
        features_list.append(max_feature(8, test, relative_difference_from_max))
    if test[9]:    
        features_list.append(mean_feature(9, test, relative_difference_from_mean))
    if test[10]:    
        features_list.append(max_feature(10, test, relative_difference_from_mean))
    if test[11]:    
        features_list.append(mean_feature(11, test, percentage_difference))
    if test[12]:    
        features_list.append(mean_feature(12, test,sd_change))
    #print("finished video", name)
    # Note: The indices are increased by 1 because the wind speed is saved at index 0 (to allow future expansion if more features are added)
    return_dict[name] = features_list



def estimate(test_videos, test, rbf_svc):
    '''
    '''
    manager = multiprocessing.Manager()
    stats_return_dict = manager.dict()
    threads = []
    exact_accuracy = 0
    lenient_accuracy = 0
    differences = []

    for video in test_videos:
        save_folder = os.path.join(directory, video)
        load_folder = os.path.join(save_folder, "DenseFlow")
        #print("Beginning estimation on video: ", video)
        
        try:
            for file in os.listdir(load_folder):
                while len(threads) >= 12:
                    for thread in threads:
                        if not thread.is_alive():
                            #print("Removing",str(thread))
                            threads.remove(thread)

                p = multiprocessing.Process(target=processVideo, args=(file, load_folder, test, stats_return_dict))
                threads.append(p)
                p.start()
        except:
            print("\n\n\n\n\nFILE NOT FOUND "+load_folder)

    for thread in threads:
        thread.join()

    for value in stats_return_dict.values():
        estimate = rbf_svc.predict([value[1:]])
        actual = value[0]
        difference = abs(estimate-actual)
        print(estimate)
        if difference<=1:
            lenient_accuracy +=1
            if difference == 0:
                exact_accuracy +=1
        differences.append(difference)
    exact_accuracy = (exact_accuracy/len(stats_return_dict)) * 100
    lenient_accuracy = (lenient_accuracy/len(stats_return_dict)) * 100
    
    return exact_accuracy, lenient_accuracy, differences


def test_order(features_number):
    '''
    '''
    all_procedures = list(itertools.product([True, False], repeat=13))
    constrained_procedure = []
    #features follow strict indexing, so true at a features index means to process it, false means to skip it.
    print(len(all_procedures))
    for test in all_procedures:
        if not test.count(True) < 12 and not test.count(True)>12:
            constrained_procedure.append(test)
    print(len(constrained_procedure))
    return constrained_procedure

def setup_test(data,test):
    new_set = []
    for values_list in data:
        new_list = []
        for value_index in range(len(values_list)):
            if test[value_index]:
                new_list.append(values_list[value_index])
        new_set.append(new_list)
    return new_set



if __name__ == '__main__':
    start = time.time()
    directory = "V:\\Uni 3\\Project\\Results\\"

    # New method: Store per wind category in lists of every feature from each clip.
    results_by_category = [[] for x in range(13)] 

    total_folders = len(os.listdir(directory))
    count = 0
    
    # do for each training set
    data_set_dir = "V:\\Uni 3\\Project\\DataSets\\"

    procedure = test_order(13)

    results = {}
    for test in procedure:
        rbfs = []
        average_exact_accuracy =0
        average_lenient_accuracy = 0
        average_differences = 0
        for data_file in os.listdir(data_set_dir):
            load_file = np.load(data_set_dir+data_file)
            # Separate into training and test datasets
            training_data = setup_test(load_file['arr_0'],test)
            training_categories = load_file['arr_1']
            test_video_paths = load_file['arr_2']
            total_test_items = len(test_video_paths)


            

            #Being testing
            print("Beginning Estimation")
            print("Total Testing videos", len(test_video_paths))
            rbfs.append(svm.SVC(kernel='rbf'))
            rbfs[-1].fit(training_data, training_categories)

            exact_accuracy, lenient_accuracy, differences = estimate(test_video_paths, test, rbfs[-1])
            set_differences = sum(differences)
            print("Overall exact accuracy",exact_accuracy)
            print("Overall lenient accuracy", lenient_accuracy)
            print("total differences", set_differences)
            print(differences)
            print("print total test videoes", total_test_items)

            output_string = "\nExact Accuracy for trail "+str(test)+":\n "+str(exact_accuracy)+"\nLenient Accuracy:"+str(lenient_accuracy)+"\nTotal differences:"+str(set_differences)
            text_output(output_string)

            average_exact_accuracy +=exact_accuracy
            average_lenient_accuracy += lenient_accuracy
            average_differences += set_differences


        average_exact_accuracy = average_exact_accuracy / 5
        average_lenient_accuracy = average_lenient_accuracy / 5
        average_differences = average_differences / 5 
        output_string = "\n\nAverage Accuracy: "+str(average_exact_accuracy)+"\nLenient Accuracy:"+str(lenient_accuracy)+ "\nAverage Total differences: "+str(average_differences)
        text_output(output_string)
        results[test] = average_exact_accuracy
    

        

    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))
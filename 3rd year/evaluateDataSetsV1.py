import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import time
import multiprocessing 
import random
from sklearn import svm
import sys
import copy
import pickle

# This file handles the creation of the training and test sets for Experiment 1
# Parts of this file may be depricated and might no longer work within the pipeline


def mean_feature(data):
    return statistics.mean(data)


def max_feature(data):
    return max(data)
    

def processVideo(file, load_folder, return_dict, debugging):
    flow_list = pickle.load( open( os.path.join(load_folder, file), "rb" ))

    mean_list = [] 
    median_list = [] 
    sd_list = []
    max_value = []
    difference_from_max = []
    difference_from_mean =[] 
    relative_difference_from_mean = [] 
    percentage_difference = []
    relative_difference_from_max = [] 
    sd_change = [] 

    if len(file.split("."))> 1:
        name = file.split(".")
    else:
        name = file


    #print("processing frames from video", file)
    for frame_index in range(len(flow_list)):
        #get optical flow from 2 frames and save changes in magnitudes
        magnitudes = [0.0]  
        step = 1

        #debugging
        if debugging:
            step = 2
        if len(flow_list[frame_index]) < 1:
            continue
        for x in range(0, len((flow_list[frame_index][0])), step):
            for y in range(0, len((flow_list[frame_index])), step):
                # Optional ignore low values || REQUIRED TO AVOID DIVIDE BY ZERO
                print(flow_list[frame_index][y][x])
                magnitude = float(np.linalg.norm(flow_list[frame_index][y][x]))
                print(magnitude)
                if magnitude > 0.001:
                    magnitudes.append(magnitude)

        mean_list.append(statistics.mean(magnitudes)) # mean of whole flow
        if len(mean_list)>1:
            difference_from_mean.append(abs(mean_list[-1] - mean_list[-2])) # Difference between this frames mean val and last frame's mean val
        median_list.append(statistics.median(magnitudes)) #median of all flow
        sd_list.append(statistics.pstdev(magnitudes)) # sd of all flow
        max_value.append(max(magnitudes)) # highest flow val

        if len(max_value)>1:
            difference_from_max.append(abs(max_value[-1] - max_value[-2])) # Difference between this frames max val and last frame's max val

        if debugging:
            # Only process up to certain number of frames (for debugging)
            # Debugging
            if frame_index > 8:
                #print("debugging; first 15 frames only.")
                break

    # For each max frame difference calculate the distance relative to mean of whole video
    for item in difference_from_max:
        relative_difference_from_max.append(item / statistics.mean(mean_list))
    
    
    #for each mean frame difference calculate the distance relative to mean of whole video
    for item in difference_from_mean:
        relative_difference_from_mean.append(item / statistics.mean(mean_list))

    
    for item_index in range(2, len(mean_list)):
        if sd_list[item_index-1] != 0:
            percentage_difference.append((mean_list[item_index]/mean_list[item_index-1])*100) 
        else:   
            percentage_difference.append(0.0) 

    
        # For every item in the standard deviation list, find the change and save it.       
        for item_index in range(2, len(sd_list)):
            sd_change.append(abs(sd_list[item_index-1]-sd_list[item_index]))
            
    video_windspeed = name.split("-")[-1]
    video_features = {} 
    
    video_features["category"] = video_windspeed
    video_features["mean"] = mean_feature(mean_list)
    video_features["median"] = mean_feature(median_list)
    video_features["sd"] = mean_feature(sd_list)
    video_features["mean_difference_from_max"] = mean_feature(difference_from_max)
    video_features["max_difference_from_max"] = max_feature(difference_from_max)
    video_features["mean_difference_from_mean"] = mean_feature(difference_from_mean)
    video_features["max_difference_from_mean"] = max_feature(difference_from_mean)
    video_features["mean_relative_difference_from_max"] = mean_feature(relative_difference_from_max)
    video_features["max_relative_difference_from_max"] = max_feature(relative_difference_from_max)
    video_features["mean_relative_difference_from_mean"] = mean_feature(relative_difference_from_mean)
    video_features["max_relative_difference_from_mean"] = max_feature(relative_difference_from_mean)
    video_features["percentage_difference"] = mean_feature(percentage_difference)
    video_features["sd_change"] = mean_feature(sd_change)


    #print("finished video", name)
    # Note: The indices are increased by 1 because the wind speed is saved at index 0 (to allow future expansion if more features are added)
    return_dict[name] = video_features


def evalSet(set_results, video_set, directory, flowtype):
    # Extract the features of each video in its own thread.
    # A maximum of 12 threads run at a time (for optimal performance on development platform) 
    count = 0
    total_folders = len(video_set)
    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for video in video_set:
        save_folder = os.path.join(directory, video)
        load_folder = os.path.join(save_folder, flowtype)

        for file in os.listdir(load_folder):
            while len(threads) >= 10:
                for thread in threads:
                    if not thread.is_alive():
                        threads.remove(thread)
                        # Return and save each feature from the video using a thread-safe dictionary 
                        for key,value in return_dict.items():
                            for feat_key, feat_val in value.items():
                                set_results[feat_key].append(feat_val) 
                            del return_dict[key]

            p = multiprocessing.Process(target=processVideo, args=(file, load_folder, return_dict, debugging))
            threads.append(p)
            p.start()

            if debugging:
                #Debugging
                #print("debugging: first file only")
                break

        count +=1 
        if count % 10 == 0:
            print("Completed {} out of {} folders".format(count, total_folders))


    # Wait for any remaining threads to finish before continuing
    for thread in threads:
        thread.join()
        for key,value in return_dict.items():
            for feat_key, feat_val in value.items():
                set_results[feat_key].append(feat_val) 
            del return_dict[key]

        
    print(len(set_results))
    print(len(set_results["category"]))

    return set_results
        


if __name__ == '__main__':
    debugging = False
    flowtype = "DenseFlow"# dense
            
    if len(sys.argv)>1:
        if sys.argv[1].lower() == "dense":
            flowtype = "DenseFlow"
        elif sys.argv[1].lower() == "points":
            flowtype = "PointsFlow"
        else:
            print(sys.argv[1])
            raise Exception("Bad argument")
        if len(sys.argv)>2:
            print(sys.argv)
            print("Debugging Mode")
            debugging = True

    start = time.time()
    

    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow")
    Save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], os.path.join("DataSets", flowtype))

    # New method: Store per wind category in lists of every feature from each clip.
    features = {
        "category" : [],
        "mean" : [], 
        "median" : [], 
        "sd" : [], 
        "mean_difference_from_max" : [], 
        "max_difference_from_max" : [], 
        "mean_difference_from_mean" : [], 
        "max_difference_from_mean" : [], 
        "mean_relative_difference_from_max" : [], 
        "max_relative_difference_from_max" : [], 
        "mean_relative_difference_from_mean" : [], 
        "max_relative_difference_from_mean" : [],
        "percentage_difference" : [],
        "sd_change" : [] 
    } 


    # Separate into training and test datasets
    training_video_sets = []
    test_video_sets = []
    
    # Get 5 different test and training configurations 
    for group in range(5):
        #Order into wind force categories
        videos_by_force = [[] for x in range(13)]
        for wind_force in range(0, 13):
            for video in os.listdir(load_directory):
                ending = "-"+str(wind_force)
                if video.endswith(ending):
                    videos_by_force[wind_force].append(video)
            
        # take 20% of the video of each wind force and separate them into a test set.
        training_video_sets.append([])
        test_video_sets.append([])
        for wind_force in range(0, 13):
            total = len(videos_by_force[wind_force])
            print(total, "videos of force", str(wind_force))
            test_set_size = int(total/5) # 20% of each force for testing
            for video in range(test_set_size):
                choice = random.choice(videos_by_force[wind_force])
                test_video_sets[group].append(choice)
                videos_by_force[wind_force].remove(choice)

            for remainder in videos_by_force[wind_force]:
                training_video_sets[group].append(remainder)
                

    total_test_videos = len(test_video_sets[0])

    print("Finished establishing data sets")

    for set_index in range(len(training_video_sets)):
        features_copy = copy.deepcopy(features)
        training_features = evalSet(features_copy, training_video_sets[set_index], load_directory, flowtype)

        features_copy = copy.deepcopy(features)
        test_features = evalSet(features_copy, test_video_sets[set_index], load_directory, flowtype)


        if not os.path.exists(os.path.join(Save_directory,str(set_index))):
            os.mkdir(os.path.join(Save_directory,str(set_index)))
        print("\n\nSaving")
        with open(Save_directory+"\\"+str(set_index)+"\\TrainingSet"+str(set_index), 'wb') as out:
            pickle.dump(training_features, out)
        with open(Save_directory+"\\"+str(set_index)+"\\TestSet"+str(set_index), 'wb') as out:
            pickle.dump(test_features, out)

    
    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))


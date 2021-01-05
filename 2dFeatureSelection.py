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
import bisect 
import numba 
import constants

# This file handles the creation of the training and test sets for Experiment 5

def mean_feature(data):
    return statistics.mean(data)

@numba.jit(nopython=True)
def upper_feature(data):
    '''
    Takes in a set of data, assumed to be sorted. 
    Returns item at the start of the top 20%
    '''
    upper_index = len(data) // 5
    return data[upper_index]
    

@numba.jit(nopython=True)
def calculateMagnitudes(vectors):

    magnitudes = numba.typed.List() 
    angles = numba.typed.List() 

    for vector in range(len(vectors)):
        magnitude = float(np.linalg.norm(vectors[vector]))

        #x not require because would be multiplied by 0
        y2 = vectors[vector][1]

        if magnitude>0:
            dot = (y2/magnitude)
            angle = np.arccos(dot)
            angles.append(angle)

        magnitudes.append(magnitude)
            
    sorted_magnitudes = sorted(magnitudes)
    return sorted_magnitudes, angles

@numba.jit(nopython=True)
def calculateVectors(track):
    x_vectors = numba.typed.List()
    y_vectors = numba.typed.List() 
    for index in range(len(track)-1):
        x_vectors.append(track[index+1][0] - track[index][0])
        y_vectors.append(track[index+1][1] - track[index][1])
    return x_vectors, y_vectors


@numba.jit(nopython=True)
def calculateRelatives( dif_fm_mean, dif_fm_sd, average_flow):
        rel_dif_fm_mean = numba.typed.List() 
        rel_dif_fm_sd  = numba.typed.List()
        for item in dif_fm_mean:
            rel_dif_fm_mean.append(item / average_flow)
        for item in dif_fm_sd:
            rel_dif_fm_sd.append(item / average_flow)
        return rel_dif_fm_mean, rel_dif_fm_sd
        


def processVideo(folder_name, load_directory, return_dict, relative = False,):
    flow_list = pickle.load( open( os.path.join(load_directory, folder_name, "Frames"), "rb" ))
    tracks_list = pickle.load( open( os.path.join(load_directory, folder_name, "Tracks"), "rb" ))

    mean_list = []

    mean_direction = []
    direction_sd = []
    sd_list = []
    dif_fm_mean = numba.typed.List()
    dif_fm_sd = numba.typed.List()
    dif_fm_mean_dir_abs = []
    dif_fm_sd_dir_abs = []

    if len(folder_name.split("."))> 1:
        name = folder_name.split(".")
    else:
        name = folder_name

    
    len_flow_list0 = len(flow_list[0])
    len_flow_list00 = len(flow_list[0][0])
    
    for frame_index in range(len(flow_list)):
        #get optical flow from 2 frames and save changes in magnitudes

        pure_vector_list = numba.typed.List()
        

        for x in range(len_flow_list0):
            for y in range(len_flow_list00):
                if flow_list[frame_index][x][y]:
                    pure_vector_list.append(np.array(flow_list[frame_index][x][y]))

        try:
            magnitudes, angles = calculateMagnitudes(pure_vector_list)
        except Exception as err:
            print(err)
            print(frame_index)
    

        frame_mean = statistics.mean(magnitudes)
        mean_list.append(frame_mean) # mean of whole flow
        
        sd_list.append(statistics.pstdev(magnitudes)) # sd of all flow

        if len(angles)>0:
            mean_direction.append(statistics.mean(angles))
            direction_sd.append(statistics.pstdev(angles))
    
    track_means = []
    for track in tracks_list:
        track_vectorsx, track_vectorsy = calculateVectors(track)
        #...
        #steal from the tracks EVD py file
        #There must be something else, right?


    average_flow = mean_feature(mean_list)

    

    #Removed from max in V3
    video_windspeed = name.split("-")[-1]
    video_features = {} 
    video_features["category"] = video_windspeed # TODO extend to metadata and make list of all metadata aspects like environement, etc
    video_features["mean"] = average_flow
    video_features["sd"] = mean_feature(sd_list) #best
    video_features["mean_direction"] = mean_feature(mean_direction) 
    video_features["direction_sd"] = mean_feature(direction_sd)


    return_dict[name] = video_features


def evalSet(opflow_directory):
    # Extract the features of each video in its own thread.
    # A maximum of 12 threads run at a time (for optimal performance on development platform) 
    start = time.time()
    set_results = {}
    count = 0
    total_folders = len(os.listdir(opflow_directory))

    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for scene in os.listdir(opflow_directory):
        load_directory = os.path.join(opflow_directory, scene)
        for folder_name in os.listdir(load_directory):
            while len(threads) >= constants.THREADS:
                for thread in threads:
                    if not thread.is_alive():
                        threads.remove(thread)
                        # Return and save each feature from the video using a thread-safe dictionary 
                        for key,value in return_dict.items():
                            for feat_key, feat_val in value.items():
                                if not feat_key in set_results.keys():
                                    set_results[feat_key] = []
                                set_results[feat_key].append(feat_val) 
                            del return_dict[key]

            p = multiprocessing.Process(target=processVideo, args=(folder_name, load_directory, return_dict))
            threads.append(p)
            p.start()


        count +=1 
        if count % 10 == 0:
            print("Completed {} out of {} folders".format(count, total_folders))
            print("Taken {} seconds so far, or approximately {} minutes.".format(time.time() - start, (time.time() - start)//60))


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
        

def inputs():
    if len(sys.argv) > 1:
        try:
            preprocessing_code = sys.argv[1]
            opflow_code = str(sys.argv[2]).split()

        except:
            print("Error in input string: using default settings")
    else:
        preprocessing_code = "4_3_500_5_3_10_C_False"
        opflow_code = "500_0.001_10_10_25_3"

    return preprocessing_code, opflow_code


if __name__ == '__main__':

    relative = False    
    
    # PLACEHOLDER - inputs are preprocessing code and opflow code
    preprocessing_code = "4_3_500_5_3_10_C_False"
    opflow_code = "500_0.001_10_10_25_3"


    start = time.time()
    
    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow", preprocessing_code, opflow_code)
    save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Datasets", preprocessing_code + "_" + opflow_code)
            

    features = evalSet(load_directory)

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    print("\n\nSaving")
    with open(os.path.join(save_directory, "2"), 'wb') as out:
        pickle.dump(features, out)


    
    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))


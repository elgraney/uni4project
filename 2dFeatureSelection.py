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
import bz2
import pickle
import _pickle as cPickle
import bisect 
import numba 
import constants
from scipy.signal import savgol_filter
import commonFunctions

# This file handles the creation of the training and test sets for Experiment 5
@numba.jit(nopython=True)
def mean_feat_nopython(data):
    return np.mean(data)


def mean_feature(data):
    if data.any():
        return mean_feat_nopython(np.array(data))
    else:
        return 0 # NOTE: using 0 is bad! Find an alternative


def sd_feat_nopython(data):
    return np.std(data)


def sd_feature(data):
    if data.any():
        return sd_feat_nopython(np.array(data))
    else:
        return 0 # NOTE: using 0 is bad! Find an alternative


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
    length = len(vectors)

    magnitudes = np.empty(length) 
    angles = np.empty(length) 

    for vector_index in range(length):
        magnitude = float(np.linalg.norm(vectors[vector_index]))

        #x not require because would be multiplied by 0
        y2 = vectors[vector_index][1]

        if magnitude>0:
            dot = (y2/magnitude)
            angle = np.arccos(dot)
            angles[vector_index]= angle
        else: 
             angles[vector_index] = 0

        magnitudes[vector_index]= magnitude
            
    sorted_magnitudes = np.sort(magnitudes)
    return sorted_magnitudes, angles


@numba.jit(nopython=True)
def angleDifference(angles):
    length = len(angles) -1
    diffs = np.empty(length) 
    for index in range(length):
        
        diffs[index] = (np.pi - abs(abs(angles[index+1] - angles[index]) - np.pi))
    return diffs


def frame_features(flow_list):
    mean_list = numba.typed.List()

    direction_sd = numba.typed.List()
    sd_list = numba.typed.List()
    
    for frame_index in range(len(flow_list)):
        #get optical flow from 2 frames and save changes in magnitudes
        flattened = flow_list[frame_index].reshape(-1, 2)

        nan_array = np.isnan(flattened)
        not_nan_array = ~ nan_array
        stripped = flattened[not_nan_array].reshape(-1,2)

        #change to unravel array
        #then strip of Nan values

        magnitudes, angles = calculateMagnitudes(stripped)

        if not magnitudes.any() or not angles.any():
            continue 

        frame_mean = mean_feature(magnitudes)
        mean_list.append(frame_mean) # mean of whole flow
        
        sd_list.append(sd_feature(magnitudes)) # sd of all flow

        if len(angles)>0:
            direction_sd.append(sd_feature(angles))
            
    return mean_list, direction_sd, sd_list



def tracks_features(tracks_list):
    track_means = []
    track_sds = []
    angle_consistency = []
    angle_range = []
    oscillation_rate = []
    oscillation_consistency = []

    #Not needed?
    #TODO remove tracks proper - not needed, handled earlier
    for track in tracks_list:
        track_proper = track[1:]
        if not len(track_proper)>1:
            continue
        
        track_vectors = [list(x) for x in track_proper]

        magnitudes, angles = calculateMagnitudes(np.array(track_vectors))
   
        if not magnitudes.any() or not angles.any():
            continue 

        angle_differences = angleDifference(angles)
        angle_consistency_addition = mean_feature(angle_differences)
        if angle_consistency_addition:
            angle_consistency.append(angle_consistency_addition)
        angle_range_addition = sd_feature(angle_differences)
        if angle_range_addition:
            angle_range.append(angle_range_addition)

        # TODO Need some way of intelligently identifying turning points
        
        if len(angle_differences) > 5:
            
            smoothed_range = savgol_filter(angle_differences, 5,3)
        else:
            smoothed_range = angle_differences

        turning_points = []
        for index in range(1, len(smoothed_range)-1):
            if smoothed_range[index] >= smoothed_range[index-1] and smoothed_range[index] >= smoothed_range[index+1]:
                #this is a maximum turn point
                if smoothed_range[index] > np.pi/5: # set the min criteria for a turning point to pi/5
                    turning_points.append(index)
        
        if turning_points:
            oscillation_rate_addition = mean_feature(np.diff(turning_points))
            if oscillation_rate_addition:
                oscillation_rate.append(oscillation_rate_addition)
            oscillation_consistency_addition = sd_feature(np.diff(turning_points))
            if oscillation_consistency_addition:
                oscillation_consistency.append(oscillation_consistency_addition)
        
        track_means.append(mean_feature(magnitudes))
        track_sds.append(sd_feature(magnitudes))

    return track_means, track_sds, angle_consistency, angle_range, oscillation_rate, oscillation_consistency



def processVideo(folder_name, load_directory, return_dict, relative = False,):
    try:
        flow_list = bz2.BZ2File(os.path.join(load_directory, folder_name, "Frames.pbz2"), "rb")
        flow_list = cPickle.load(flow_list)
        tracks_list = bz2.BZ2File(os.path.join(load_directory, folder_name, "Tracks.pbz2"), "rb")
        tracks_list = cPickle.load(tracks_list)


    except:
        print("failed to load file {}".format(os.path.join(load_directory, folder_name)))
        return

    if len(folder_name.split("."))> 1:
        name = folder_name.split(".")
    else:
        name = folder_name

    mean_list, direction_sd, sd_list = frame_features(flow_list)
    track_means, track_sds, angle_consistency, angle_range, oscillation_rate, oscillation_consistency = tracks_features(tracks_list)

    #NOTE: these feats are combined in a bad way! averaging removes significant trends! FIND A BETTER WAY!
    video_windspeed = name.split("-")[-1]
    video_features = {} 
    video_features["category"] = video_windspeed # TODO extend to metadata and make list of all metadata aspects like environement, etc
    video_features["mean"] = mean_feature(np.array(mean_list))
    video_features["meanSD"] = sd_feature(np.array(mean_list))
    video_features["sd"] = mean_feature(np.array(sd_list)) #best
    video_features["sdSD"] = sd_feature(np.array(sd_list))
    video_features["dirSd"] = mean_feature(np.array(direction_sd))
    video_features["dirSdSD"] = sd_feature(np.array(direction_sd))
    video_features["trMeans"] = mean_feature(np.array(track_means))
    video_features["trMeansSD"] = sd_feature(np.array(track_means))
    video_features["trSds"] = mean_feature(np.array(track_sds))
    video_features["trSdsSD"] = sd_feature(np.array(track_sds))
    video_features["aglCons"] = mean_feature(np.array(angle_consistency))
    video_features["aglConsSD"] = sd_feature(np.array(angle_consistency))
    video_features["aglRng"] = mean_feature(np.array(angle_range))
    video_features["aglRngSD"] = sd_feature(np.array(angle_range))
    video_features["oscRate"] = mean_feature(np.array(oscillation_rate))
    video_features["oscRateSD"] = sd_feature(np.array(oscillation_rate))
    video_features["oscCons"] = mean_feature(np.array(oscillation_consistency)) 
    video_features["oscConsSD"] = sd_feature(np.array(oscillation_consistency)) 

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

    return set_results
        


if __name__ == '__main__':
    start = time.time()

    preprocessing_code, opflow_code, filename = commonFunctions.code_inputs(sys.argv)
    
    
    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow", preprocessing_code, opflow_code)
    save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Datasets", preprocessing_code, opflow_code)

    commonFunctions.makedir(os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Datasets", preprocessing_code))

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    else:
        print("Save directory already exists")
        exit()


    features = evalSet(load_directory)

    print("\n\nSaving")
    with open(os.path.join(save_directory, filename), 'wb') as out:
        pickle.dump(features, out)


    
    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))


    #CURRENT RUN TIME APPROX 10182 (10 threads)
    #Approx same with nopython mean. Try same with sd.
    #Reduced to 7543 with 10 threads and sd
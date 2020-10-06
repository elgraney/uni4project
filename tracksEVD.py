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

# This file handles the creation of the training and test sets for Experiment 6; tracks optical flow

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
def calculateMagnitudes(track, debugging = False):
    magnitudes = numba.typed.List() 
    angles = numba.typed.List() 
    for vector_index in range(len(track)):
        vector = track[vector_index]

        magnitude = float(np.linalg.norm(vector))
        #x not require because would be multiplied by 0
        y1 = 1
        y2 = vector[1]
            
        if magnitude>0:
            # calculate angle from north 
            dot = y1 * (y2/magnitude)
            angle = np.arccos(dot)
            angles.append(angle)
        magnitudes.append(magnitude)

    return magnitudes, angles


def processVideo(file, load_folder, return_dict, debugging = False):
    tracks = pickle.load( open( os.path.join(load_folder, file), "rb" ))

    new_tracks = numba.typed.List()
    for track in tracks: # remove single or empty tracks
        if len(track)>1:
            new_tracks.append(np.array(track))

    
    tracks = new_tracks
    if len(file.split("."))> 1:
        name = file.split(".")
    else:
        name = file

    mean = []
    sd = []
    val_range = []
    rel_sd = []
    most_change = []
    average_direction = []
    oscillation_rate = []
    oscillation_consistency = []
    mean_oscillation_range = []
    oscillation_range_spread = []
    mean_direction_changes = []

    for track in tracks:
        #get optical flow from 2 frames and save changes in magnitudes
    
        magnitudes, angles = calculateMagnitudes(track)

        if len(angles) < 1: # No useful motion
            continue

        mean.append(statistics.mean(magnitudes))
        sd.append(statistics.pstdev(magnitudes))
        val_range.append(max(magnitudes)-min(magnitudes))
        rel_sd.append(sd[-1]/val_range[-1])
        most_change.append(max(magnitudes))

        average_direction.append(statistics.mean(angles))

        direction_changes = []
        turn_points = []
        
        oscillation_range = []

        for angle_index in range(len(angles)):
            if angle_index > 0:
                change = abs(angles[angle_index] - angles[angle_index-1])
                direction_changes.append(change)
                if change > (np.pi/2): # if there is over a 90 degree change of direction
                    turn_points.append(angle_index)
        

        if  len(turn_points) == 0:
            #no oscillation: Do not attempt remaining features
            continue

        mean_direction_changes.append(statistics.mean(direction_changes))
        oscillation_rate.append(len(turn_points) / len(track)) 

        Frames_between_changes = []
        for index in range(len(turn_points)):
            if index > 0:
                Frames_between_changes.append(turn_points[index] - turn_points[index-1])
        if len(Frames_between_changes)<2:
            Frames_between_changes.append((len(track)/2) +1) # presumably oscillation happens beyond the range of the clip

        oscillation_consistency.append(statistics.pstdev(Frames_between_changes)) # spread of time between changes = how consistently it changes (smaller spread = more consistent)

        
        for index in range(len(turn_points)):
            if index > 0:
                frame_index1 = turn_points[index-1]
                frame_index2 = turn_points[index]

                total = 0
                for mag_index in range(frame_index1, frame_index2):
                    total += magnitudes[mag_index]
                oscillation_range.append(total)
        if len(Frames_between_changes)<2:
            oscillation_range.append(magnitudes[turn_points[0]] - min(magnitudes))
        mean_oscillation_range.append(statistics.mean(oscillation_range))
        oscillation_range_spread.append(statistics.pstdev(oscillation_range))
    
    if len(mean) == 0 or len(turn_points) == 0:
        #no useful data to return.
        return
    video_windspeed = name.split("-")[-1]
    video_features = {} 
    video_features["category"] = video_windspeed
    video_features["mean"] = mean_feature(mean)
    video_features["sd"] = mean_feature(sd)
    video_features["value range"] = mean_feature(val_range)
    video_features["Relative sd"] = mean_feature(rel_sd)
    video_features["most_change"] = mean_feature(most_change)
    video_features["average_direction"] = mean_feature(average_direction) 
    video_features["average_direction_change"] = mean_feature(mean_direction_changes)
    video_features["oscillation_rate"] = mean_feature(oscillation_rate)
    video_features["oscillation_consistency"] = mean_feature(oscillation_consistency)
    video_features["mean_oscillation_range"] = mean_feature(mean_oscillation_range)
    video_features["oscillation_range_spread"] = mean_feature(oscillation_range_spread)
    
    return_dict[name] = video_features

def evalSet(video_set, directory, flowtype, start_time):
    # Extract the features of each video in its own thread.
    # A maximum of 12 threads run at a time (for optimal performance on development platform) 
    set_results = {}
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
                                if not feat_key in set_results.keys():
                                    set_results[feat_key] = []
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
        
            current_time = time.time()
            time_taken = current_time - start_time
            print("Taken {} seconds so far, or approximately {} minutes.".format(time_taken, time_taken//60))
   


    # Wait for any remaining threads to finish before continuing
    for thread in threads:
        thread.join()
        for key,value in return_dict.items():
            for feat_key, feat_val in value.items():
                set_results[feat_key].append(feat_val) 
            del return_dict[key]
        

    return set_results
        

if __name__ == '__main__':

    flowtype = "DenseFlow"# dense

    debugging = False     
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

    start = time.time()
    
    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow")
    Save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], os.path.join("DataSets", flowtype))

    # New method: Store per wind category in lists of every feature from each clip.

    # Separate into training and test datasets
    training_video_sets = []
    test_video_sets = []
    
    # Get 5 different test and training configurations 
    for group in range(3):
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
            if group == 0:
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
    
        training_features = evalSet(training_video_sets[set_index], load_directory, flowtype, start)

        test_features = evalSet(test_video_sets[set_index], load_directory, flowtype, start)


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


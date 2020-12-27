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
def calculateMagnitudes(flow_list, frame_index, debugging = False):
    step = 1
    #debugging
    if debugging:
        step = 2
    magnitudes = numba.typed.List() 
    angles = numba.typed.List() 

    for x in range(0, len((flow_list[frame_index][0])), step):
        for y in range(0, len((flow_list[frame_index])), step):
            magnitude = float(np.linalg.norm(flow_list[frame_index][y][x]))
            #x not require because would be multiplied by 0
            y1 = 1
            y2 = flow_list[frame_index][y][x][1]
            
            if magnitude>0:
                dot = y1 * (y2/magnitude)

                angle = np.arccos(dot)
                angles.append(angle)

            magnitudes.append(magnitude)
            

    sorted_magnitudes = sorted(magnitudes)
    return sorted_magnitudes, angles

@numba.jit(nopython=True)
def calculateRelatives( dif_fm_mean, dif_fm_sd, average_flow):
        rel_dif_fm_mean = numba.typed.List() 
        rel_dif_fm_sd  = numba.typed.List()
        for item in dif_fm_mean:
            rel_dif_fm_mean.append(item / average_flow)
        for item in dif_fm_sd:
            rel_dif_fm_sd.append(item / average_flow)
        return rel_dif_fm_mean, rel_dif_fm_sd
        


def processVideo(file, load_folder, return_dict, relative = False, debugging = False):
    flow_list = pickle.load( open( os.path.join(load_folder, file), "rb" ))

    typed_flow_list = numba.typed.List()
    [typed_flow_list.append(np.array(x)) for x in flow_list]
    flow_list = typed_flow_list

    mean_list = []
    median_list =[]
    mean_direction = []
    direction_sd = []
    sd_list = []
    upper_mark = []
    dif_fm_mean = numba.typed.List()
    dif_fm_sd = numba.typed.List()
    dif_fm_mean_dir_abs = []
    dif_fm_sd_dir_abs = []


    if len(file.split("."))> 1:
        name = file.split(".")
    else:
        name = file

    for frame_index in range(len(flow_list)):
        #get optical flow from 2 frames and save changes in magnitudes
        
        if len(flow_list[frame_index]) < 1:
            continue

        magnitudes, angles = calculateMagnitudes(flow_list, frame_index, debugging)

        mean = statistics.mean(magnitudes)
        mean_list.append(mean) # mean of whole flow
        median_list.append(statistics.median(magnitudes)) #median of all flow
        sd_list.append(statistics.pstdev(magnitudes)) # sd of all flow
        if len(angles)>0:
            mean_direction.append(statistics.mean(angles))
            direction_sd.append(statistics.pstdev(angles))
        if len(mean_list)>1:
            dif_fm_mean.append(abs(mean_list[-1] - mean_list[-2])) # Difference between this fmames mean val and last frame's mean val
        if len(sd_list)>1:
            dif_fm_sd.append(abs(sd_list[-1] - sd_list[-2])) # Difference between this frames sd val and last frame's sd val
        if len(mean_direction) > 1:
            dif_fm_mean_dir_abs.append(abs(mean_direction[-1] - mean_direction[-2]))
        if len(direction_sd) > 1:
            dif_fm_sd_dir_abs.append(abs(direction_sd[-1] - direction_sd[-2]))

        upper_mark.append(upper_feature(np.array(magnitudes)))
       
    # For each max frame difference calculate the distance relative to mean of whole video
    average_flow = mean_feature(mean_list)
    

    #Are We testing relative? If so do the following. If not leave commented.
    if relative:
        dif_fm_mean, dif_fm_sd = calculateRelatives( dif_fm_mean, dif_fm_sd, average_flow)

    # Rate of change of mean & sd:
    mean_change_rate = []
    sd_change_rate = []
    for index in range(1, len(dif_fm_mean)):
    
        mean_change_rate.append(abs(dif_fm_mean[index] - dif_fm_mean[index-1]))
        sd_change_rate.append(abs(dif_fm_sd[index] - dif_fm_sd[index-1]))

    #Removed from max in V3
    video_windspeed = name.split("-")[-1]
    video_features = {} 
    video_features["category"] = video_windspeed
    video_features["mean"] = average_flow
    video_features["median"] = mean_feature(median_list)
    video_features["sd"] = mean_feature(sd_list) #best
    video_features["mean_dif_fm_mean"] = mean_feature(dif_fm_mean)
    video_features["max_dif_fm_mean"] = max(dif_fm_mean)
    video_features["dif_fm_sd"] = mean_feature(dif_fm_sd)
    video_features["rate_of_mean_change"] = mean_feature(mean_change_rate)
    video_features["rate_of_sd_change"] = mean_feature(sd_change_rate)
    video_features["mean_direction"] = mean_feature(mean_direction) 
    video_features["direction_sd"] = mean_feature(direction_sd)
    video_features["dir_dif_fm_mean_abs"] = mean_feature(dif_fm_mean_dir_abs)
    video_features["dir_dif_fm_sd_abs"] = mean_feature(dif_fm_sd_dir_abs)

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
        
    print(len(set_results))
    print(len(set_results["category"]))

    return set_results
        

if __name__ == '__main__':

    flowtype = "DenseFlow"# dense
    relative = False
    debugging = False     
    if len(sys.argv)>1:
        if sys.argv[1].lower() == "dense":
            flowtype = "DenseFlow"
        elif sys.argv[1].lower() == "points":
            flowtype = "PointsFlow"
        else:
            print(sys.argv[1])
            raise Exception("Bad argument; argument 1")

        if len(sys.argv)>2:
            if sys.argv[2].lower() == "true":
                relative = True
            elif sys.argv[2].lower() == "false":
                relative = False
            else:
                print(sys.argv[1])
                raise Exception("Bad argument; argument 2")


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


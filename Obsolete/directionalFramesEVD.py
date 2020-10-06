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

def mean_feature(data):
    return statistics.mean(data)

def max_feature(data):
    return max(data)
    
@numba.jit(nopython=True)
def arrangeFlow(flow_list, frame_index, debugging = False):
    '''
    Take a nested list of optical flow vectors
    Reformat to flattened list
    separate x and y movement
    '''
    flow = np.reshape(flow_list[frame_index], (-1, 2))
    flattened_flow = flow.flatten()
    x = flattened_flow[::2]
    y = flattened_flow[1::2]
    return x, y


@numba.jit(nopython=True)
def calculateRelatives(dif_fm_mean, dif_fm_sd, x_average_flow, y_average_flow):
    '''
    Divide each statistic by mean to make relative. 
    '''
    rel_dif_fm_mean = numba.typed.List() 
    rel_dif_fm_sd  = numba.typed.List()

    for item in dif_fm_mean:
        flow_dif = (item[0] / x_average_flow, item[1] / y_average_flow)
        rel_dif_fm_mean.append(flow_dif)
    for item in dif_fm_sd:
        rel_dif_fm_sd.append((item[0] / x_average_flow, item[1] / y_average_flow))

    return rel_dif_fm_mean, rel_dif_fm_sd

def xyFrameFeatures(flow):
    '''
    Calculate mean, median, sd for a list of 1D optical flow values
    '''
    mean = statistics.mean(flow) # mean of whole flow
    median= statistics.median(flow) #median of all flow
    sd = statistics.pstdev(flow) # sd of all flow
    return mean, median, sd


def processVideo(file, load_folder, return_dict, relative = False, debugging = False):
    '''
    Load a sequence of optical flow arrays for pairs of frames in a clip
    Record statistics for each optical flow array
    Calculate stats for the optical flow of the entire clip
    '''
    flow_list = pickle.load( open( os.path.join(load_folder, file), "rb" ))

    typed_flow_list = numba.typed.List()
    [typed_flow_list.append(np.array(x)) for x in flow_list] # USING HOTFIX: resolve in optical flow!
    flow_list = typed_flow_list

    # Initialise variables as numba typed lists to enable their use in numba 'nopython' functions
    x_mean_list = numba.typed.List()
    y_mean_list = numba.typed.List()
    x_median_list = numba.typed.List()
    y_median_list = numba.typed.List()
    x_sd_list = numba.typed.List()
    y_sd_list = numba.typed.List()
    dif_fm_mean = numba.typed.List()
    dif_fm_sd = numba.typed.List()

    # split to remove file extension
    if len(file.split("."))> 1:
        name = file.split(".")
    else:
        name = file

    for frame_index in range(len(flow_list)):
        #get optical flow from between 2 frames
        if len(flow_list[frame_index]) < 1:
            continue

        x, y = arrangeFlow(flow_list, frame_index)
        x = x.astype(float)
        y = y.astype(float)

        mean, median, sd = xyFrameFeatures(x)
        x_mean_list.append(mean), x_median_list.append(median), x_sd_list.append(sd)
        mean, median, sd = xyFrameFeatures(y)
        y_mean_list.append(mean), y_median_list.append(median), y_sd_list.append(sd)

        val = numba.typed.List()
        if len(x_mean_list)>1:
            val.append(abs(x_mean_list[-1] - x_mean_list[-2]))
            val.append(abs(y_mean_list[-1] - y_mean_list[-2]))
            dif_fm_mean.append(val) # Difference between this fmames mean val and last frame's mean val
        val = numba.typed.List()
        if len(x_sd_list)>1:
            val.append(abs(x_sd_list[-1] - x_sd_list[-2]))
            val.append(abs(y_sd_list[-1] - y_sd_list[-2]))
            dif_fm_sd.append(val) # Difference between this frames sd val and last frame's sd val

    # For get average mean, median, sd for x and y
    average_x_flow = mean_feature(x_mean_list)
    average_y_flow = mean_feature(y_mean_list)

    average_median_x_flow = mean_feature(x_median_list)
    average_median_y_flow = mean_feature(y_median_list)

    average_sd_x_flow = mean_feature(x_sd_list)
    average_sd_y_flow = mean_feature(y_sd_list)
    #Are We testing relative? If so do the following. If not leave commented.
    if relative:
        dif_fm_mean, dif_fm_sd = calculateRelatives(dif_fm_mean, dif_fm_sd, average_x_flow, average_y_flow)
    
    av_x_dif_fm_mean = mean_feature(vector[0] for vector in dif_fm_mean)
    av_y_dif_fm_mean = mean_feature(vector[1] for vector in dif_fm_mean)

    av_x_dif_fm_sd = mean_feature(vector[0] for vector in dif_fm_sd)
    av_y_dif_fm_sd = mean_feature(vector[1] for vector in dif_fm_sd)


    video_windspeed = name.split("-")[-1]
    video_features = {} 
    
    video_features["category"] = video_windspeed
    video_features["mean"] = [average_x_flow, average_y_flow]
    video_features["median"] = [average_median_x_flow, average_median_y_flow]
    video_features["sd"] = [average_sd_x_flow, average_sd_y_flow]
    video_features["mean_dif_fm_mean"] = [av_x_dif_fm_mean, av_y_dif_fm_mean]
    video_features["mean_dif_fm_sd"] =  [av_x_dif_fm_sd, av_y_dif_fm_sd]



    #print("finished video", name)
    # Note: The indices are increased by 1 because the wind speed is saved at index 0 (to allow future expansion if more features are added)
    return_dict[name] = video_features


def evalSet(video_set, directory, flowtype):
    '''
    Handle threading to run feature calculation for each video set
    '''
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
    # handle input arguments
    flowtype = "DenseFlow"
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

            if len(sys.argv)>3:
                print(sys.argv)
                print("Debugging Mode")
                debugging = True


    start = time.time()
    
    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow")
    Save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], os.path.join("DataSets", flowtype))

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
    
        training_features = evalSet(training_video_sets[set_index], load_directory, flowtype)

        test_features = evalSet(test_video_sets[set_index], load_directory, flowtype)


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


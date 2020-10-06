

import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import time
import multiprocessing 
import random

def average_per_category(data):
    '''
    Finds the average of the given data for each wind force
    '''
    averages = [statistics.mean(x) for x in data]
    return averages
    

def processVideo(file, load_folder, return_dict):
    flow_list = np.load(os.path.join(load_folder, file))
    mean_list = []
    median_grouped_list = []
    sd_list = []
    max_value = []
    difference_from_max = []
    difference_from_mean =[]
    sd_change = []
    relative_difference_from_mean = []
    relative_difference_from_max = []
    percentage_difference = []

    name = file.split(".")[0]

    print("processing frames from video", file)
    for frame_index in range(len(flow_list)):
        
        #get optical flow from 2 frames and save changes in magnitudes
        magnitudes = [0.0]  
        for x in range(len((flow_list[0][0]))):
            for y in range(len((flow_list[0]))):
                # Optional ignore low values || REQUIRED TO AVOID DIVIDE BY ZERO
                magnitude = float(np.linalg.norm(flow_list[frame_index][y][x]))
                if magnitude > 0.001:
                    magnitudes.append(magnitude)

        mean_list.append(statistics.mean(magnitudes)) # mean of whole flow
        median_grouped_list.append(statistics.median_grouped(magnitudes)) #median of all flow
        sd_list.append(statistics.pstdev(magnitudes)) # sd of all flow
        max_value.append(max(magnitudes)) # highest flow val

        if len(max_value)>1:
            difference_from_max.append(abs(max_value[-1] - max_value[-2])) # Difference between this frames max val and last frame's max val
        if len(mean_list)>1:
            difference_from_mean.append(abs(mean_list[-1] - mean_list[-2])) # Difference between this frames mean val and last frame's mean val

        # Only process up to certain number of frames (for debugging)
        '''
        if frame_index > 8:
            print("debugging; first 20 frames only.")
            break
        '''
        
    print("finished processing frames from video", name)
    # For each max frame difference calculate the distance relative to mean of whole video
    for item in difference_from_max:
        relative_difference_from_max.append(item / statistics.mean(mean_list))
    #for each mean frame difference calculate the distance relative to mean of whole video
    for item in difference_from_mean:
        relative_difference_from_mean.append(item / statistics.mean(mean_list))

    # For every item in the standard deviation list, find the change and save it.       
    for item_index in range(2, len(sd_list)):
        sd_change.append(abs(sd_list[item_index]-sd_list[item_index-1]))

    # Calc percentage change relative to the mean  of last frame
    for item_index in range(2, len(mean_list)):
        if sd_list[item_index-1] != 0:
            percentage_difference.append((mean_list[item_index]/mean_list[item_index-1])*100) 
        else:   
            percentage_difference.append(0.0) 

    video_windspeed = name.split("-")[-1]

    mean = statistics.mean(mean_list)
    med_gr = statistics.mean(median_grouped_list)
    sd = statistics.mean(sd_list)
    mean_dif_from_max = statistics.mean(difference_from_max)
    max_dif_from_max = max(difference_from_max)
    sd_change = statistics.mean(sd_change)
    rel_dif_from_max = statistics.mean(relative_difference_from_max)
    max_rel_dif_from_max = max(relative_difference_from_max)
    perc_dif = statistics.mean(percentage_difference)

    mean_dif_from_mean = statistics.mean(difference_from_mean)
    max_dif_from_mean = max(difference_from_mean)
    rel_dif_from_mean = statistics.mean(relative_difference_from_mean)
    max_rel_dif_from_mean = max(relative_difference_from_mean)

    return_dict[name] = (int(video_windspeed), mean, med_gr,sd,mean_dif_from_max,max_dif_from_max, sd_change,rel_dif_from_max,perc_dif, max_rel_dif_from_max, mean_dif_from_mean, max_dif_from_mean,rel_dif_from_mean,max_rel_dif_from_mean)


def best_fit_forces(test_val, training_list):
    # alter to return top 3 forces
    least_dif = 9999
    forces = []
    force = 0
    for loop in range(3):
        least_dif = 9999
        for index in range(len(training_list)):
            difference = abs(training_list[index] - test_val)
            if difference < least_dif and  not index in forces:
                least_dif = difference
                force = index
        forces.append(force)      
    return forces


def output_estimation(video, test_stats, training_stats, accuracy, differences):
    '''
    Takes the stats for a single test video and the stats for the training set and finds the most appropriate category for the test video
    params:
        video: string name of the video
        test_stats: 1D list of the features for a single test video, with each feature at a specific index.
        training_stats: 2D list; one list for each feature. Each feature list hold 13 items; the average for the feature for each force.
        return_dict: threading compatable storage for output data.
    '''
    #current results aren't great: lets try getting a top three and ranking them appropriately
    actual = test_stats[0]
    mean_est = best_fit_forces(test_stats[1], training_stats[0])
    med_gr_est = best_fit_forces(test_stats[2], training_stats[1])
    sd_est = best_fit_forces(test_stats[3], training_stats[2])
    mean_dif_from_max_est = best_fit_forces(test_stats[4], training_stats[3])
    max_dif_from_max_est = best_fit_forces(test_stats[5], training_stats[4])
    mean_sd_ch_est = best_fit_forces(test_stats[6], training_stats[5])
    mean_rel_dif_from_max_est = best_fit_forces(test_stats[7], training_stats[6])
    perc_dif_est = best_fit_forces(test_stats[8], training_stats[7])
    max_rel_dif_from_max_est = best_fit_forces(test_stats[9], training_stats[8])

    mean_dif_from_mean_est = best_fit_forces(test_stats[10], training_stats[9])
    max_dif_from_mean_est = best_fit_forces(test_stats[11], training_stats[10])
    rel_dif_from_mean_est = best_fit_forces(test_stats[12], training_stats[11])
    max_rel_dif_from_mean_est = best_fit_forces(test_stats[13], training_stats[12])

    print("Printing results for",video)
    print(actual,":",mean_est,med_gr_est,sd_est,mean_dif_from_max_est,max_dif_from_max_est, mean_sd_ch_est, mean_rel_dif_from_max_est, perc_dif_est, max_rel_dif_from_max_est, mean_dif_from_mean_est, mean_dif_from_mean_est, max_dif_from_mean_est, rel_dif_from_mean_est, max_rel_dif_from_mean_est)
    #all_results = mean_est+med_gr_est+sd_est+mean_dif_from_max_est+max_dif_from_max_est+mean_sd_ch_est+mean_rel_dif_from_max_est+perc_dif_est+max_rel_dif_from_max_est+mean_dif_from_mean_est+mean_dif_from_mean_est+max_dif_from_mean_est+rel_dif_from_mean_est+max_rel_dif_from_mean_est
    #all_results = perc_dif_est
    all_results = sd_est+mean_dif_from_max_est+max_dif_from_max_est+mean_sd_ch_est+mean_rel_dif_from_max_est+max_rel_dif_from_max_est#+mean_dif_from_mean_est+mean_dif_from_mean_est+max_dif_from_mean_est+rel_dif_from_mean_est+max_rel_dif_from_mean_est

    try:
        prediction = statistics.mode(all_results)
        #prediction = all_results[0]
        print(prediction)
    except:
        print("no clear mode, printing mean")
        prediction = statistics.mean(all_results)
        print(prediction)

    prediction_difference = abs(prediction - actual)
    if prediction_difference < 1.5:
        print("prediction was correct")
        accuracy +=1
        
    else:
        print("prediction was incorrect")
    return accuracy, prediction_difference


def estimate(test_videos, stats):
    '''
    '''
    manager = multiprocessing.Manager()
    stats_return_dict = manager.dict()
    accuracy = 0
    differences = []

    for video in test_videos:
        save_folder = os.path.join(directory, video)
        load_folder = os.path.join(save_folder, "DenseFlow")
        print("Beginning estimation on video: ", video)

        file = os.listdir(load_folder)[0]
        while len(threads) >= 12:
            for thread in threads:
                if not thread.is_alive():
                    print("Removing",str(thread))
                    threads.remove(thread)

        p = multiprocessing.Process(target=processVideo, args=(file, load_folder, stats_return_dict))
        threads.append(p)
        p.start()

    for thread in threads:
        thread.join()
    for key,value in stats_return_dict.items():
        accuracy, difference = output_estimation(key, value, stats, accuracy, differences)
        differences.append(difference)
    return accuracy, differences



if __name__ == '__main__':
    start = time.time()
    directory = "V:\\Uni 3\\Project\\Results\\"

    #setup storage for every feature, organised into the 13 wind force categories
    mean_mean_list = [[] for x in range(13)]
    mean_median_grouped_list = [[] for x in range(13)]
    mean_sd_list = [[] for x in range(13)]

    mean_difference_from_max = [[] for x in range(13)]
    max_difference_from_max = [[] for x in range(13)]

    mean_difference_from_mean = [[] for x in range(13)]
    max_difference_from_mean = [[] for x in range(13)]

    mean_relative_difference_from_max = [[] for x in range(13)]
    max_relative_difference_from_max = [[] for x in range(13)]

    mean_relative_difference_from_mean = [[] for x in range(13)]
    max_relative_difference_from_mean = [[] for x in range(13)]

    mean_percentage_difference = [[] for x in range(13)]
    mean_sd_change = [[] for x in range(13)]

    print("Seperating test and training sets")
    total_folders = len(os.listdir(directory))
    print(total_folders,"folders to process")
    count = 0

    # Separate into training and test datasets
    training_videos = []
    test_videos = []
    #Order into wind force categories
    videos_by_force = [[] for x in range(13)]
    for wind_force in range(0, 13):
        for video in os.listdir(directory):
            ending = "-"+str(wind_force)
            if video.endswith(ending):
                videos_by_force[wind_force].append(video)

    # take 20% of the video of each wind force and separate them into a test set.
    for wind_force in range(0, 13):
        total = len(videos_by_force[wind_force])
        print(total, "videos of force", str(wind_force))
        test_set_size = int(total/5) # 20% of each force for testing
        for video in range(test_set_size):
            choice = random.choice(videos_by_force[wind_force])
            test_videos.append(choice)
            videos_by_force[wind_force].remove(choice)
        for remainder in videos_by_force[wind_force]:
            training_videos.append(remainder)

    total_test_videos = len(test_videos)

    print("Finished establishing data sets")

    # Extract the features of each video in its own thread.
    # A maximum of 12 threads run at a time (for optimal performance on development platform) 
    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for video in training_videos:
        save_folder = os.path.join(directory, video)
        load_folder = os.path.join(save_folder, "DenseFlow")

        for file in os.listdir(load_folder):
            while len(threads) >= 12:
                for thread in threads:
                    if not thread.is_alive():
                        print("thread",str(thread),"is finished")
                        threads.remove(thread)
                        
                        # Return and save each feature from the video using a thread-safe dictionary 
                        for key,value in return_dict.items():
                            mean_mean_list[value[0]].append(value[1])
                            mean_median_grouped_list[value[0]].append(value[2])
                            mean_sd_list[value[0]].append(value[3])
                            mean_difference_from_max[value[0]].append(value[4])
                            max_difference_from_max[value[0]].append(value[5])
                            mean_sd_change[value[0]].append(value[6])
                            mean_relative_difference_from_max[value[0]].append(value[7])
                            mean_percentage_difference[value[0]].append(value[8])
                            max_relative_difference_from_max[value[0]].append(value[9])

                            mean_difference_from_mean[value[0]].append(value[10])
                            max_difference_from_mean[value[0]].append(value[11])
                            mean_relative_difference_from_mean[value[0]].append(value[12])
                            max_relative_difference_from_mean[value[0]].append(value[13])
                            print("Deleting",key)
                            del return_dict[key]

            p = multiprocessing.Process(target=processVideo, args=(file, load_folder, return_dict))
            threads.append(p)
            p.start()


            print("debugging: first file only")
            break
        
        print("Passed ", video, "folder")
        print("Approximately ", str(int((count/total_folders)*100))+"% complete")
        count += 1


    # Wait for any remaining threads to finish before continuing
    for thread in threads:
        thread.join()
        for value in return_dict.values():
            mean_mean_list[value[0]].append(value[1])
            mean_median_grouped_list[value[0]].append(value[2])
            mean_sd_list[value[0]].append(value[3])
            mean_difference_from_max[value[0]].append(value[4])
            max_difference_from_max[value[0]].append(value[5])
            mean_sd_change[value[0]].append(value[6])
            mean_relative_difference_from_max[value[0]].append(value[7])
            mean_percentage_difference[value[0]].append(value[8])
            max_relative_difference_from_max[value[0]].append(value[9])

            mean_difference_from_mean[value[0]].append(value[10])
            max_difference_from_mean[value[0]].append(value[11])
            mean_relative_difference_from_mean[value[0]].append(value[12])
            max_relative_difference_from_mean[value[0]].append(value[13])



    # 1. Produce a function for each means of estimation
    # 2. Get the stats from the test set
    # 3. Run stats against the training stats and do majority voting for wind force.

    stats_list = []
    stats_list.append(average_per_category(mean_mean_list))
    stats_list.append(average_per_category(mean_median_grouped_list))
    stats_list.append(average_per_category(mean_sd_list))
    stats_list.append(average_per_category(mean_difference_from_max))
    stats_list.append(average_per_category(max_difference_from_max))
    stats_list.append(average_per_category(mean_sd_change))
    stats_list.append(average_per_category(mean_relative_difference_from_max))
    stats_list.append(average_per_category(mean_percentage_difference))
    stats_list.append(average_per_category(max_relative_difference_from_max))
    stats_list.append(average_per_category(mean_difference_from_mean))
    stats_list.append(average_per_category(max_difference_from_mean))
    stats_list.append(average_per_category(mean_relative_difference_from_mean))
    stats_list.append(average_per_category(max_relative_difference_from_mean))

    #Being testing
    print("Beginning Estimation")
    print("Total Testing videos", len(test_videos))
    accuracy, differences = estimate(test_videos, stats_list)
    overall_accuracy = (accuracy/total_test_videos) * 100
    print("Overall accuracy",overall_accuracy)
    print("total differences", sum(differences))
    print(differences)
    print("print total test videoes", total_test_videos)

    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))
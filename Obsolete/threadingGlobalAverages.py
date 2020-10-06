

import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import time
import multiprocessing 



def text_output(data, title):
    with open('V:\\Uni 3\\Project\\output.txt','a') as result_file:
        force = 0
        result_file.write("\n"+title+"\n")
        for item in data:
            result_file.write(str(force)+"\n")

            result_file.writelines(str(statistics.mean(item)))
            result_file.write("\n")
            force+=1
    result_file.close()

def plot_statistic(data, title):
    forces = [x for x in range(0,13)]
    averages = [statistics.mean(x) for x in data]
    plt.plot(forces, averages)

    plt.savefig(title+".png")
    plt.clf()

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
        change = abs(sd_list[item_index-1]-sd_list[item_index])
        sd_change.append(change)
        # Also work out percentage change relative to the last frame for standard deviation change
        if sd_list[item_index-1] != 0:
            percentage_difference.append((change / sd_list[item_index-1])*100) 
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


    


if __name__ == '__main__':
    start = time.time()
    directory = "V:\\Uni 3\\Project\\Results\\"

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

    total = len(os.listdir(directory))
    print(total,"folders to process")
    count = 0

    threads = []
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    for video in os.listdir(directory):
        save_folder = os.path.join(directory, video)
        load_folder = os.path.join(save_folder, "DenseFlow")

        for file in os.listdir(load_folder):
            while len(threads) >= 12:
                for thread in threads:
                    if not thread.is_alive():
                        print("thread",str(thread),"is finished")
                        threads.remove(thread)

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

            print("Set to first video only")
            break

        print("Approximately ", str(int((count/total)*100))+"% complete")
        count += 1
        print("Passed ", video, "folder")


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


    text_output(mean_mean_list, "Mean")
    text_output(mean_median_grouped_list, "Median grouped")
    text_output(mean_sd_list, "Standard Deviation")
    text_output(mean_difference_from_max, "Mean Difference from Max")
    text_output(max_difference_from_max, "Max Difference from max")
    text_output(mean_sd_change, "Standard Deviation Change")
    text_output(mean_relative_difference_from_max, "Mean Difference Relative to Mean from max")
    text_output(mean_percentage_difference, "Mean Percentage Change")
    text_output(max_relative_difference_from_max, "Max Difference Relative to Mean from max")
    text_output(mean_difference_from_mean, "Mean Difference from Mean")
    text_output(max_difference_from_mean, "Max Difference from Mean")
    text_output(mean_relative_difference_from_mean, "Mean Difference Relative to Mean from Mean")
    text_output(max_relative_difference_from_mean, "Max Difference Relative to Mean from Mean")

    plot_statistic(mean_mean_list, "Mean")
    plot_statistic(mean_median_grouped_list, "Median grouped")
    plot_statistic(mean_sd_list, "Standard Deviation")
    plot_statistic(mean_difference_from_max, "Mean Difference from Max")
    plot_statistic(max_difference_from_max, "Max Difference from max")
    plot_statistic(mean_sd_change, "Standard Deviation Change")
    plot_statistic(mean_relative_difference_from_max, "Mean Difference Relative to Mean from max")
    plot_statistic(mean_percentage_difference, "Mean Percentage Change")
    plot_statistic(max_relative_difference_from_max, "Max Difference Relative to Mean from max")
    plot_statistic(mean_difference_from_mean, "Mean Difference from Mean")
    plot_statistic(max_difference_from_mean, "Max Difference from Mean")
    plot_statistic(mean_relative_difference_from_mean, "Mean Difference Relative to Mean from Mean")
    plot_statistic(max_relative_difference_from_mean, "Max Difference Relative to Mean from Mean")

    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))
import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import time


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


start = time.time()
dir = "V:\\Uni 3\\Project\\Results\\"


mean_mean_list = [[] for x in range(13)]
mean_median_grouped_list = [[] for x in range(13)]
mean_sd_list = [[] for x in range(13)]

mean_difference = [[] for x in range(13)]
max_difference = [[] for x in range(13)]

mean_relative_difference = [[] for x in range(13)]
mean_percentage_difference = [[] for x in range(13)]

mean_sd_change = [[] for x in range(13)]



for video in os.listdir(dir):
    save_folder = os.path.join(dir, video)
    load_folder = os.path.join(save_folder, "DenseFlow")


    for file in os.listdir(load_folder):
        print("Processing ",file,"...")
        flow_list = np.load(os.path.join(load_folder, file))
        mean_list = [0.001]
        median_grouped_list = [0.001]
        sd_list = [0.001]
        max_value = [0.001]
        difference = [0.001]
        sd_change = [0.001]
        relative_difference = [0.001]
        percentage_difference = [0.001]

        for frame_index in range(len(flow_list)):
            print("processing frame",str(frame_index+1))
            magnitudes = [0.0]

            for x in range(len((flow_list[0][0]))):
                for y in range(len((flow_list[0]))):
                    # Optional ignore low values || REQUIRED TO AVOID DIVIDE BY ZERO
                    magnitude = float(np.linalg.norm(flow_list[frame_index][y][x]))
                    if magnitude > 0.01:
                        magnitudes.append(magnitude)

            mean_list.append(statistics.mean(magnitudes))
            median_grouped_list.append(statistics.median_grouped(magnitudes))
            sd_list.append(statistics.pstdev(magnitudes))
            max_value.append(max(magnitudes))

            difference.append(abs(max_value[-1] - max_value[-2]))

            for item_index in range(1, len(sd_list)):
                change = abs(sd_list[item_index-1]-sd_list[item_index])
                sd_change.append(change)
                if sd_list[item_index-1] != 0:
                    percentage_difference.append((change / sd_list[item_index-1])*100) 
                else:   
                    percentage_difference.append(0.0) 
            
            if frame_index > 8:
                print("debugging; first 20 frames only")
                break
            

        for item in difference:
            relative_difference.append(item / statistics.mean(mean_list))
            

        video_windspeed = video.split("-")[-1]

        mean_mean_list[int(video_windspeed)].append(statistics.mean(mean_list))
        mean_median_grouped_list[int(video_windspeed)].append(statistics.mean(median_grouped_list))
        mean_sd_list[int(video_windspeed)].append(statistics.mean(sd_list))

        mean_difference[int(video_windspeed)].append(statistics.mean(difference))
        max_difference[int(video_windspeed)].append(max(difference) - min(difference))

        mean_sd_change[int(video_windspeed)].append(statistics.mean(sd_change))

        mean_relative_difference[int(video_windspeed)].append(statistics.mean(relative_difference))

        mean_percentage_difference[int(video_windspeed)].append(statistics.mean(percentage_difference))


        print("First Video Only")
        break
        




text_output(mean_mean_list, "Mean")
text_output(mean_median_grouped_list, "Median grouped")
text_output(mean_sd_list, "Standard Deviation")
text_output(mean_difference, "Mean Difference")
text_output(max_difference, "Max Difference")
text_output(mean_sd_change, "Standard Deviation Change")
text_output(mean_relative_difference, "Mean Difference Relative to Mean")
text_output(mean_percentage_difference, "Mean Percentage Change")

plot_statistic(mean_mean_list, "Mean")
plot_statistic(mean_median_grouped_list, "Median grouped")
plot_statistic(mean_sd_list, "Standard Deviation")
plot_statistic(mean_difference, "mean difference")
plot_statistic(max_difference, "Max difference")
plot_statistic(mean_sd_change, "Standard Deviation Change")
plot_statistic(mean_relative_difference, "Mean Difference Relative to Mean")
plot_statistic(mean_percentage_difference, "Mean Percentage Change")


end=time.time()
print("Moment of truth...")
print("standard global averages time:")
print(str(end - start))
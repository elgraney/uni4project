import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import time
import multiprocessing 
import random

def text_output(text):
    with open('V:\\Uni 3\\Project\\accuracy.txt','a') as result_file:
        result_file.write(text)
    result_file.close()


def average_per_category(data):
    '''
    Finds the average of the given data for each wind force
    '''
    averages = None
    try:
        averages = [statistics.mean(x) for x in data]
        return averages
    except:
        print("No items in given set, passing")
        #return None
    

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
        #Debugging step 2 NOT CURRENTLY IN
        for x in range(0, len((flow_list[0][0])), 2):
            for y in range(0, len((flow_list[0])), 2):
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
        if frame_index > 7:
            print("debugging; first 30 frames only.")
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

    features_list.append(mean_feature(0, test, mean_list))
    features_list.append(mean_feature(1, test, median_grouped_list))
    features_list.append(mean_feature(2, test, sd_list))
    features_list.append(mean_feature(3, test, difference_from_max))
    features_list.append(max_feature(4, test, difference_from_max))
    features_list.append(mean_feature(5, test, difference_from_mean))
    features_list.append(max_feature(6, test, difference_from_mean))
    features_list.append(mean_feature(7, test, relative_difference_from_max))
    features_list.append(max_feature(8, test, relative_difference_from_max))
    features_list.append(mean_feature(9, test, relative_difference_from_mean))
    features_list.append(max_feature(10, test, relative_difference_from_mean))
    features_list.append(mean_feature(11, test, percentage_difference))
    features_list.append(mean_feature(12, test,sd_change))
    print("finished video", name)
    # Note: The indices are increased by 1 because the wind speed is saved at index 0 (to allow future expansion if more features are added)
    return_dict[name] = features_list


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


def output_estimation(video, test_stats, training_stats, exact_accuracy, lenient_accuracy, differences):
    '''
    Takes the stats for a single test video and the stats for the training set and finds the most appropriate category for the test video
    params:
        video: string name of the video
        test_stats: 1D list of the features for a single test video, with the value of each feature at a specific index.
        training_stats: 2D list; one list for each feature. Each feature list hold 13 items; the average for the feature for each force.
        return_dict: threading compatable storage for output data.
    '''
    # SELF NOTE current results aren't great: lets try getting a top three and ranking them appropriately
    actual = test_stats[0]
    estimate_list = []
    # training stats follows normal index positioning, test stats has wind force inserted at the front, bumping all indices by 1
    for index in range(len(training_stats)):
        if not training_stats[index] is None:
            estimate_list += best_fit_forces(test_stats[index+1], training_stats[index])
        else:
            print("index", index, "value is None. Skipping estimation")

    print("Printing results for",video)
    print(actual,":", estimate_list)
    
    try:
        prediction = statistics.mode(estimate_list)
        #prediction = estimate_list...
    except:
        #print("no clear mode, printing mean")
        if len(estimate_list) == 3:
            prediction = estimate_list[0]
            #print("Single estimate, estimate is first choice")
        else:
            prediction = statistics.mean(estimate_list)

    prediction_difference = abs(prediction - actual)

    print("Prediction for", video,":", prediction)
    if prediction_difference < 0.5:
        #print("prediction was correct")
        exact_accuracy +=1
    if prediction_difference <1.5:
        lenient_accuracy +=1
    else:
        pass
        #print("prediction was incorrect")
    return exact_accuracy, lenient_accuracy, prediction_difference


def estimate(test_videos, stats, test):
    '''
    '''
    manager = multiprocessing.Manager()
    stats_return_dict = manager.dict()
    exact_accuracy = 0
    lenient_accuracy = 0
    differences = []

    for video in test_videos:
        save_folder = os.path.join(directory, video)
        load_folder = os.path.join(save_folder, "DenseFlow")
        #print("Beginning estimation on video: ", video)

        file = os.listdir(load_folder)[0]
        while len(threads) >= 12:
            for thread in threads:
                if not thread.is_alive():
                    #print("Removing",str(thread))
                    threads.remove(thread)

        p = multiprocessing.Process(target=processVideo, args=(file, load_folder, test, stats_return_dict))
        threads.append(p)
        p.start()

    for thread in threads:
        thread.join()
    for key,value in stats_return_dict.items():
        exact_accuracy, lenient_accuracy, difference = output_estimation(key, value, stats, exact_accuracy, lenient_accuracy, differences)
        differences.append(difference)
    return exact_accuracy, lenient_accuracy, differences

def test_order(features_number):
    '''
    '''
    procedure = [] # setup list to contain booleans representing each feature. 
    #features follow strict indexing, so true at a features index means to process it, false means to skip it.
    for index in range(1, features_number):
        order = [False for x in range(features_number)]
        order[index] = True
        procedure.append(order)
    return procedure

if __name__ == '__main__':
    start = time.time()
    directory = "V:\\Uni 3\\Project\\Results\\"

    #setup storage for every feature, organised into the 13 wind force categories
    '''
    Throughout the program the indexes of the features are kept the same. 
    Each of the following are the mean values of that feature for every data item in each wind category
    0: The mean value (for each clip) of mean optical flow magnitudes between each two frames.
    1: The mean value (for each clip) of the grouped median of the optical flow maginitudes between each two frames.
    2: The mean value (for each clip) of the standard deviation of the optical flow magnitudes between each two frames.
    3: The mean value (for each clip) of the maximum difference of the optical flow magnitudes between each two frames.
    4: The maximum value (for each clip) of the maximum difference of the optical flow magnitudes between each two frames.
    5: The mean value (for each clip) of the mean difference of the optical flow magnitudes between each two frames.
    6: The maximum value (for each clip) of the mean difference of the optical flow magnitudes between each two frames.
    7: The mean value (for each clip) of the maximum difference, relative to the mean optical flow magnitude of the clip, of the optical flow magnitudes between each two frames.
    8: The maximum value (for each clip) of the maximum difference, relative to the mean optical flow magnitude of the clip, of the optical flow magnitudes between each two frames.
    9: The mean value (for each clip) of the mean difference, relative to the mean optical flow magnitude of the clip, of the optical flow magnitudes between each two frames.
    10: The maximum value (for each clip) of the mean difference, relative to the mean optical flow magnitude of the clip, of the optical flow magnitudes between each two frames.
    11: The mean value (for each clip) of the mean percentage difference of the mean optical flow of the last two frames and the previous two frames.
    12: The mean value (for each clip) of the mean standard deviation change between the standard deviation of optical flow of the the last two frames and the previous two frames.
    '''
    features = [[[] for x in range(13)] for y in range(13)]


    print("Seperating test and training sets")
    total_folders = len(os.listdir(directory))
    print(total_folders,"folders to process")
    count = 0

    # Separate into training and test datasets
    training_video_sets = []
    test_video_sets = []
    
    # Get 3 different test and training configurations 
    for group in range(3):
        #Order into wind force categories
        videos_by_force = [[] for x in range(13)]
        for wind_force in range(0, 13):
            for video in os.listdir(directory):
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

            '''
            #Debugging
            limiter = 10
            count = 0
            '''

            for remainder in videos_by_force[wind_force]:
                training_video_sets[group].append(remainder)
                
                '''
                #Debugging
                count +=1
                if count >= limiter:
                    print("LIMITED")
                    break
                '''

    total_test_videos = len(test_video_sets[0])

    print("Finished establishing data sets")

    # Set up feature test order
    procedure = test_order(13) # there are currently 13 unique features that can be tested

    

    for test in procedure:
        
        average_exact_accuracy = 0
        average_lenient_accuracy = 0
        average_differences = 0
        text_output(("\n\nStarting test" + str(test) + "\n"))
        for set_index in range(len(test_video_sets)):
            features = [[[] for x in range(13)] for y in range(13)]
            # Extract the features of each video in its own thread.
            # A maximum of 12 threads run at a time (for optimal performance on development platform) 
            threads = []
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            for video in training_video_sets[set_index]:
                save_folder = os.path.join(directory, video)
                load_folder = os.path.join(save_folder, "DenseFlow")

                
                #debuggin
                file_no = 0
                

                for file in os.listdir(load_folder):
                    
                    file_no +=1
                    #debugging
                    if file_no % 2 == 0:
                        pass
                    else:
                    
                        while len(threads) >= 12:
                            for thread in threads:
                                if not thread.is_alive():
                                    #print("thread",str(thread),"is finished")
                                    threads.remove(thread)
                                        
                                    # Return and save each feature from the video using a thread-safe dictionary 
                                    for key,value in return_dict.items():
                                        for item_index in range(len(test)):
                                            # Only save features being considered in current test
                                            if test[item_index]: 
                                                features[item_index][value[0]].append(value[item_index+1])
                                                # we use value[item_index+1] because the wind force has been inserted at index 0, bumping the features up one
                                            else:
                                                #Debugging
                                                #print("Skipping index", item_index, ": feature not in test.")
                                                pass

                                        #print("Deleting",key)
                                        del return_dict[key]

                        p = multiprocessing.Process(target=processVideo, args=(file, load_folder, test, return_dict))
                        threads.append(p)
                        p.start()

                        '''   
                        #Debugging
                        print("debugging: first file only")
                        break
                        '''
                        
                
                print("Passed ", video, "folder")
                print("Approximately ", str(int((count/total_folders)*100))+"% complete")
                count += 1


            # Wait for any remaining threads to finish before continuing
            for thread in threads:
                thread.join()
                for value in return_dict.values():
                    for item_index in range(len(test)):
                        # Only save features being considered in current test
                        if test[item_index]: # If that index is True that feature is being considered in this test
                            features[item_index][value[0]].append(value[item_index+1])
                        else:
                            #Debugging
                            #print("Skipping index", item_index, ": feature not in test.")
                            pass



            # 1. Produce a function for each means of estimation
            # 2. Get the stats from the test set
            # 3. Run stats against the training stats and do majority voting for wind force.

            # Calculate and save average of each feature for each category of wind
            print("\n\n\nFUCK\n\n\n")
            
            
            stats_list = []
            for feature in features:
                for force in feature:
                    #print(len(force))
                    print(force)
                stats_list.append(average_per_category(feature))
            print("features stats")

            '''
            print("DEBUGGING")
            print("training stats")
            print(stats_list)
            '''


            #Being testing
            print("Beginning Estimation")
            print("Total Testing videos", len(test_video_sets[set_index]))
            exact_accuracy, lenient_accuracy, differences = estimate(test_video_sets[set_index], stats_list, test)
            set_exact_accuracy = (exact_accuracy/total_test_videos) * 100
            set_lenient_accuracy = (lenient_accuracy/total_test_videos) * 100
            set_differences = sum(differences)
            print("Overall exact accuracy",set_exact_accuracy)
            print("Overall lenient accuracy", set_lenient_accuracy)
            print("total differences", set_differences)
            print(differences)
            print("print total test videoes", total_test_videos)

            output_string = "\nExact Accuracy for trail "+str(set_index)+": "+str(set_exact_accuracy)+"\nLenient Accuracy:"+str(set_lenient_accuracy)+"\nTotal differences:"+str(set_differences)
            text_output(output_string)

            average_exact_accuracy +=set_exact_accuracy
            average_lenient_accuracy += set_lenient_accuracy
            average_differences += set_differences

            '''
            #Debugging
            average_exact_accuracy = average_exact_accuracy*3
            average_differences = average_differences*3
            print("Debugging: Doing only 1 set")
            break
            '''

        average_exact_accuracy = average_exact_accuracy / 3
        average_lenient_accuracy = average_lenient_accuracy / 3
        average_differences = average_differences / 3 
        output_string = "\n\nAverage Accuracy: "+str(average_exact_accuracy)+"\nLenient Accuracy:"+str(set_lenient_accuracy)+ "\nAverage Total differences: "+str(average_differences)
        text_output(output_string)
        

    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))
import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import time
import multiprocessing 
import random
from sklearn import svm

def text_output(text):
    with open('V:\\Uni 3\\Project\\accuracy.txt','a') as result_file:
        result_file.write(text)
    result_file.close()


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
        #Debugging step 2  CURRENTLY IN
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
        if frame_index > 18:
            print("debugging; first 15 frames only.")
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
    #print("finished video", name)
    # Note: The indices are increased by 1 because the wind speed is saved at index 0 (to allow future expansion if more features are added)
    return_dict[name] = features_list



def estimate(test_videos, test, rbf_svc):
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

    for value in stats_return_dict.values():
        estimate = rbf_svc.predict([value[1:]])
        actual = value[0]
        difference = abs(estimate-actual)
        print(estimate)
        if difference<=1:
            lenient_accuracy +=1
            if difference == 0:
                exact_accuracy +=1
        differences.append(difference)
    return exact_accuracy, lenient_accuracy, differences

def test_order(features_number):
    '''
    '''
    procedure = [[True for x in range(features_number)]] # setup list to contain booleans representing each feature. 
    #features follow strict indexing, so true at a features index means to process it, false means to skip it.
    for index in range(1, features_number):
        order = [True for x in range(features_number)]
        order[index] = False
        procedure.append(order)
    return procedure

if __name__ == '__main__':
    start = time.time()
    directory = "V:\\Uni 3\\Project\\Results\\"

    # New method: Store per wind category in lists of every feature from each clip.
    results_by_category = [[] for x in range(13)] 

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
    
    rbfs = []

    for test in procedure:
        
        average_exact_accuracy = 0
        average_lenient_accuracy = 0
        average_differences = 0
        text_output(("\n\nStarting test" + str(test) + "\n"))
        for set_index in range(len(test_video_sets)):
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
                                        results_by_category[value[0]].append(value[1:]) 
                        
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
                        
                        
                
                #print("Passed ", video, "folder")
                print("Approximately ", str(int((count/total_folders)*100))+"% complete")
                count += 1


            # Wait for any remaining threads to finish before continuing
            for thread in threads:
                thread.join()
                for key,value in return_dict.items():
                    results_by_category[value[0]].append(value[1:]) 


            # 1. Produce a function for each means of estimation
            # 2. Get the stats from the test set
            # 3. Run stats against the training stats and do majority voting for wind force.
            

            data_list = []
            category_list =[]
            for category_index in range(len(results_by_category)):
                for result in results_by_category[category_index]:
                    data_list.append(result)
                    category_list.append(category_index)


            #debugging
            print("LENGTHS")
            print(len(data_list))
            print(len(category_list))

            #Being testing
            print("Beginning Estimation")
            print("Total Testing videos", len(test_video_sets[set_index]))
            rbfs.append(svm.SVC(kernel='rbf'))
            rbfs[-1].fit(data_list, category_list)

            exact_accuracy, lenient_accuracy, differences = estimate(test_video_sets[set_index], test, rbfs[-1])
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
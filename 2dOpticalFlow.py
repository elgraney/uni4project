import cv2
import numpy as np
import os
import multiprocessing
import pickle
import sys
import shutil
import time



# This file handles the calculation, formatting and saving of both dense and points frame wise optical flow

def optical_flow(frame_dir, flow_folder, subscene, feature_params, lk_params):
    #start = time.time()
    index = 0
    detect_interval = 1
    tracks = []
    complete_tracks = []
    list_dir = os.listdir(frame_dir)
    len_list_dir = len(list_dir)

    while(index < len_list_dir-1):
        frame = cv2.imread(os.path.join(frame_dir,list_dir[index]))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray

            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) # find new points
            p0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params) #do in reverse
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1 # keep points that are the same forward and back
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag: #track has ended
                    complete_tracks.append(tr)
                    continue # skip points that aren't the same both ways
                if x > frame.shape[1] or y > frame.shape[0]:
                    complete_tracks.append(tr)
                    continue # end tracks that go beyond the resolution
                tr.append((x, y))
                new_tracks.append(tr)
                if (index == len_list_dir-2): #last frame
                    complete_tracks.append(tr)
            tracks = new_tracks


        if index % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([index, (x, y)])

        index += 1
        prev_gray = frame_gray

    # Dense tracks:
    height, width, _ = frame.shape 

    framewise_tracks = [[[None for k in range(height)] for j in range(width)] for i in range(len_list_dir-2)] # [index][x][y]

    for track in complete_tracks:
        index = track[0]
        for track_index in range(1, len(track)-1):
            x = track[track_index][0]
            y = track[track_index][1]
            dx, dy = track[track_index+1][0]-x, track[track_index+1][1]-y

            framewise_tracks[index+track_index-1][int(x)][int(y)] = [dx, dy]

    # TODO: How do I test this is working correctly?
  
    with open(os.path.join(flow_folder, "TF_"+subscene), 'wb') as fp:   
        pickle.dump(framewise_tracks, fp)
    #print(time.time()-start)


def inputs():
    if len(sys.argv) > 2:
        try:
            var1 = sys.argv[2]
        except:
            print("Error in input string: using default settings")
    # vars
    return



if __name__ == "__main__":
    start=time.time()

    #TODO input string of preprocessing args
    # Hardcoded:
    preprocessing_code = "4_3_500_5_3_10_C_False"

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 500,
                        qualityLevel = 0.001,
                        minDistance = 10,
                        blockSize = 10 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (25,25),
                    maxLevel = 3,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
   
    opflow_code = "{}_{}_{}_{}_{}_{}".format(str(feature_params["maxCorners"]),
                                            str(feature_params["qualityLevel"]), 
                                            str(feature_params["minDistance"]), 
                                            str(feature_params["blockSize"]), 
                                            str(lk_params["winSize"][0]), 
                                            str(lk_params["maxLevel"]))

    replace = True

    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Frames", preprocessing_code)

    save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow", preprocessing_code)

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    save_directory = os.path.join(save_directory, opflow_code)
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    
    threads=[]

    total_folders = len(os.listdir(load_directory))
    current_folder = 0
    for scene in os.listdir(load_directory):
        
        scenes_directory = os.path.join(load_directory, scene)
        save_folder = os.path.join(save_directory, scene)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        elif len(os.listdir(save_folder)) == 0:
            shutil.rmtree(save_folder)
            os.mkdir(save_folder)
        else:
            if replace:
                shutil.rmtree(save_folder)
                os.mkdir(save_folder)
            else: 
                print("path {} already filled, skipping".format(save_folder))
                continue

        for subscene in os.listdir(scenes_directory):
            frame_dir = os.path.join(scenes_directory,subscene)
                   
            while len(threads) >= 12:
                for thread in threads:
                    if not thread.is_alive():
                        threads.remove(thread)
              
            p1 = multiprocessing.Process(target=optical_flow, args=(frame_dir, save_folder, subscene, feature_params, lk_params))

            threads.append(p1)
            p1.start() 

        current_folder +=1 
        if current_folder % round(total_folders/100) == 0:
            print("Completed {} out of {} folders".format(current_folder, total_folders))
        
    print(time.time()-start)
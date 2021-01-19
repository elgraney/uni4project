import cv2
import constants
import numpy as np
import os
import multiprocessing
import pickle
import sys
import shutil
import time
import commonFunctions
import numba
from numba import vectorize
from scipy.ndimage import affine_transform



# This file handles the calculation, formatting and saving of both dense and points frame wise optical flow


def camera_motion_negation(prev_pts, curr_pts):
    #Find transformation matrix
    m = cv2.estimateAffine2D(prev_pts, curr_pts) 

    transform = m[0]
    inverse_padded_transform = np.vstack((cv2.invertAffineTransform(transform),np.array([0.,0.,1.])))#invertAffineTransform outputs a 3x2 matrix: add [0,0,1] row to make 3x3.
    return inverse_padded_transform


@numba.jit(nopython=True)
def calculate_vectors(track_vectors, inv_transforms):

    corrected_vectors = np.empty((len(track_vectors),2))
    for (x, y), transform, index in zip(track_vectors, inv_transforms, range(len(track_vectors))):
        
        vector = np.array([x, y, 0.]).reshape(3,1)
        res = np.dot(np.ascontiguousarray(transform), vector) # does dot apply the inverse transform correctly?
        corrected_vectors[index] = [res[0][0], res[1][0]]
    return corrected_vectors


def minus_tracks(list1, list2):
    return np.subtract(list1, list2)


def format_track(track, framewise_tracks, transforms):
    index = track[0]
    list1 = np.array(track[2:])
    list2 = np.array(track[1:-1])
    difference_list = minus_tracks(list1, list2)
    for vector_index in range(len(difference_list)):

        x = difference_list[vector_index][0]

        y = difference_list[vector_index][1]
        framewise_tracks[index+vector_index][int(x)][int(y)] = [x, y]

    # Tracks
    track_vector = calculate_vectors(difference_list, np.array(transforms[index:index+len(track)]))

    return framewise_tracks, track_vector




def optical_flow(frame_dir, flow_folder, subscene, feature_params, lk_params):
    #start = time.time()
    index = 0
    detect_interval = 1
    tracks = []
    complete_tracks = []
    list_dir = os.listdir(frame_dir)
    len_list_dir = len(list_dir)

    # params for ShiTomasi corner detection
    camera_feature_params = dict( maxCorners = 20,
                    qualityLevel = 0.01,
                    minDistance = 20,
                    blockSize = 3 )

    transforms = []

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

            sp0 = cv2.goodFeaturesToTrack(frame_gray, **camera_feature_params)
            sp1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, sp0, None, **lk_params) # find new points

            transforms.append(camera_motion_negation(sp0, sp1)) # currently piggybacking on tracks points --> not ideal, switch to different selector

            new_tracks = []
            for tr,(x,y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag: #track has ended
                    complete_tracks.append(tr)
                    continue # skip points that aren't the same both ways
                if x > frame.shape[1] or y > frame.shape[0]:
                    complete_tracks.append(tr)
                    continue # end tracks that go beyond the resolution
                tr.append(np.array([x, y]))
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
    framewise_tracks = np.full((len_list_dir-2, width, height, 2), np.nan)

    adjusted_tracks_list =[]
    for track in complete_tracks:
        if len(track) > 2:
            framewise_tracks, adjusted_track_vectors = format_track(track, framewise_tracks, transforms)
            adjusted_tracks_list.append(adjusted_track_vectors)

    # TODO: How do I test this is working correctly?
    commonFunctions.makedir(os.path.join(flow_folder, subscene))
    
    with open(os.path.join(flow_folder, subscene, "Frames"), 'wb') as fp:   
        pickle.dump(framewise_tracks, fp)
    with open(os.path.join(flow_folder, subscene, "Tracks"), 'wb') as fp:   
        pickle.dump(adjusted_tracks_list, fp)



def inputs():
    if len(sys.argv) > 1:
        try:
            preprocessing_code = sys.argv[1]
            opflow_code = str(sys.argv[2]).split()
            replace = sys.argv[3]
            maxCorners = opflow_code[0]
            qualityLevel = opflow_code[1]
            minDistance = opflow_code[2]
            blockSize = opflow_code[3]
            winSize = opflow_code[4]
            maxLevel = opflow_code[5]

        except:
            print("Error in input string: using default settings")
    else:
        preprocessing_code = "4_3_500_5_3_10_C_False"
        opflow_code = "500_0.001_10_10_25_3"
        maxCorners = 500
        qualityLevel = 0.001
        minDistance = 10
        blockSize = 10
        winSize = 25
        maxLevel = 3
        replace = False # MAKE FALSE AGAIN 

    return preprocessing_code, opflow_code, replace, maxCorners, qualityLevel, minDistance, blockSize, winSize, maxLevel



if __name__ == "__main__":
    start=time.time()

    #TODO input string of preprocessing args
    # Hardcoded:
    preprocessing_code = "4_3_500_5_3_10_C_False"
    preprocessing_code, opflow_code, replace, maxCorners, qualityLevel, minDistance, blockSize, winSize, maxLevel = inputs()

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = maxCorners,
                        qualityLevel = qualityLevel,
                        minDistance = minDistance,
                        blockSize = blockSize )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (winSize,winSize),
                    maxLevel = maxLevel,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
   

  

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
                   
            while len(threads) >= constants.THREADS:
                for thread in threads:
                    if not thread.is_alive():
                        threads.remove(thread)
              
            p1 = multiprocessing.Process(target=optical_flow, args=(frame_dir, save_folder, subscene, feature_params, lk_params))

            threads.append(p1)
            p1.start() 

        current_folder +=1 
        if current_folder % round(total_folders/100) == 0:
            print("Completed {} out of {} folders in {} seconds, or {} minutes".format(current_folder, total_folders, time.time()-start, (time.time()-start)/60 ))
        
    print(time.time()-start)
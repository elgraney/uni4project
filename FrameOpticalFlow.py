import cv2
import numpy as np
import os
import multiprocessing
import pickle
import sys
import shutil

# This file handles the calculation, formatting and saving of both dense and points frame wise optical flow

def process_clip_dense(frame_dir, flow_folder, subscene):
    '''
    calculate dense flow from a series of frames
    '''
    try: 
        flow_list = []
        for file_index in range(1, len(os.listdir(frame_dir))):
            frame1 = cv2.imread(os.path.join(frame_dir, os.listdir(frame_dir)[file_index-1]))
            frame2 = cv2.imread(os.path.join(frame_dir, os.listdir(frame_dir)[file_index]))

            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            flow_list.append(flow)

        with open(os.path.join(flow_folder, "DF_"+subscene), 'wb') as fp:  
            pickle.dump(flow_list, fp)
    except Exception as err:
        print("Processing failed on clip {} with error {}".format(subscene, err))      

def process_clip_points(frame_dir, flow_folder, subscene):
    '''
    calculate points flow from a series of frames
    '''
    index = 0
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 500,
                        qualityLevel = 0.001,
                        minDistance = 10,
                        blockSize = 10 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (25,25),
                    maxLevel = 3,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    detect_interval = 1
    tracks = []

    flow_list = [] #flow is a frame by frame map of all optical flow values. (not necessarily in order)

    while(index < len(os.listdir(frame_dir))-1):
        frame = cv2.imread(os.path.join(frame_dir, os.listdir(frame_dir)[index]))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_flow = []
        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) # find new points
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params) #do in reverse
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1 # keep points that are the same forward and back
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag: 
                    continue # skip points that aren't the same both ways
                (x0, y0) = tr[-1]
                tr.append((x, y)) 
                dx, dy = x-x0, y-y0
                frame_flow.append([dx,dy])
                new_tracks.append(tr)
            if len(frame_flow)>0:
                flow_list.append([frame_flow])
            tracks = new_tracks

        if index % detect_interval == 0:
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)

            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])

        index += 1
        prev_gray = frame_gray
    with open(os.path.join(flow_folder, "PF_"+subscene), 'wb') as fp: 
        pickle.dump(flow_list, fp)


if __name__ == "__main__":
    
    #handle input args
    flowtype = 0 # dense
    load = "Frames"
    replace = False
    if len(sys.argv)>1:
        if sys.argv[1].lower() == "dense":
            flowtype = 0
        elif sys.argv[1].lower() == "points":
            flowtype = 1
        else:
            print(sys.argv[1])
            raise Exception("Bad argument")
        if len(sys.argv)>2:
            load = sys.argv[2]
            if len(sys.argv)>3 and sys.argv[3].lower() == "true":
                replace = True

    try:
        load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], load)
    except Exception as err:
        raise Exception("Bad argument; argument 2, invalid folder")

    save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow")
    threads=[]

    total_folders = len(os.listdir(load_directory))
    current_folder = 0

    for scene in os.listdir(load_directory):
        
        scenes_directory = os.path.join(load_directory, scene)
        save_folder = os.path.join(save_directory, scene)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if flowtype == 0:
            flow_folder = os.path.join(save_folder, "DenseFlow")
        else:
            flow_folder = os.path.join(save_folder, "PointsFlow")
            
        if os.path.exists(flow_folder) and replace == True:
            shutil.rmtree(flow_folder)
        elif os.path.exists(flow_folder) and replace == False:
            print("Contains {} folder already, skipping".format(flowtype))
            continue

        os.mkdir(flow_folder)
        for subscene in os.listdir(scenes_directory):
            frame_dir = os.path.join(scenes_directory,subscene)
                   
            while len(threads) >= 12:
                for thread in threads:
                    if not thread.is_alive():
                        threads.remove(thread)
            if flowtype ==0:    
                p1 = multiprocessing.Process(target=process_clip_dense, args=(frame_dir,flow_folder, subscene))
            else:
                p1 = multiprocessing.Process(target=process_clip_points, args=(frame_dir,flow_folder, subscene))
            threads.append(p1)
            p1.start() 

        current_folder +=1 
        if current_folder % round(total_folders/100) == 0:
            print("Completed {} out of {} folders".format(current_folder, total_folders))

    
import cv2
import numpy as np
import os
import multiprocessing
import pickle
import sys
import shutil

# This file handles the calculation, formatting and saving of tracks optical flow

def process_clip_points(frame_dir, flow_folder, subscene):
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
    complete_tracks = []

    while(index < len(os.listdir(frame_dir))-1):
        frame = cv2.imread(os.path.join(frame_dir, os.listdir(frame_dir)[index]))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params) # find new points
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params) #do in reverse
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1 # keep points that are the same forward and back
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag: #track has ended
                    complete_tracks.append(tr)
                    continue # skip points that aren't the same both ways
                tr.append((x, y)) 
                new_tracks.append(tr)
                if (index == len(os.listdir(frame_dir))-2): #last frame
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
                    tracks.append([(x, y)])

        index += 1
        prev_gray = frame_gray

    dif_tracks = []
    for track in tracks:
        dif_track = []
        for index in range(len(track)-1):
            dx, dy = track[index+1][0]-track[index][0], track[index+1][1]-track[index][1]
            dif_track.append([dx, dy])
        dif_tracks.append(dif_track)
    
    with open(os.path.join(flow_folder, "TF_"+subscene), 'wb') as fp: 
        pickle.dump(dif_tracks, fp)


if __name__ == "__main__":
    
    #MAKE INPUT ARG FOR DENSE OR POINTS
    flowtype = 0 # dense
    replace = False
    load = "Frames"
    if len(sys.argv)>1:
        if sys.argv[1].lower() == "dense":
            flowtype = 0
        elif sys.argv[1].lower() == "points":
            flowtype = 1
        elif sys.argv[1].lower() == "tracks":
            flowtype = 2
        else:
            print(sys.argv[1])
            raise Exception("Bad argument")
        if len(sys.argv)>2:
            load = sys.argv[2]
            if len(sys.argv)>3 and sys.argv[3].lower() == "true":
                replace = True



    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], load)
    save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow")
    threads=[]

    total_folders = len(os.listdir(load_directory))
    current_folder = 0

    for scene in os.listdir(load_directory):
        #print("Processing",scene,"folder")
        
        scenes_directory = os.path.join(load_directory, scene)
        save_folder = os.path.join(save_directory, scene)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if flowtype == 0:
            flow_folder = os.path.join(save_folder, "DenseFlow")
        if flowtype == 2:
            flow_folder = os.path.join(save_folder, "TracksFlow")
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

                   
            while len(threads) >= 10:
                for thread in threads:
                    if not thread.is_alive():
                        threads.remove(thread)
            p1 = multiprocessing.Process(target=process_clip_points, args=(frame_dir,flow_folder, subscene))
            threads.append(p1)
            p1.start() 
                        

        current_folder +=1 
        if current_folder % round(total_folders/100) == 0:
            print("Completed {} out of {} folders".format(current_folder, total_folders))

    
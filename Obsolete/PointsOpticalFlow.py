import cv2
import numpy as np
import os
import multiprocessing

def process_clip(frame_dir, flow_folder, subscene):
    frames_folder = os.path.join(flow_folder, "PointsFlow_"+subscene)
    os.mkdir(frames_folder)

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


    track_len = 10
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
                frame_flow.append(dx)
                frame_flow.append(dy)
                new_tracks.append(tr)
            
            frame_flow = np.reshape(frame_flow,(-1, 2))
            flow_list.append(frame_flow)
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
    for frame_index in range(len(flow_list)):
        np.save(os.path.join(frames_folder, "PF_"+str(frame_index)+subscene), flow_list[frame_index])

if __name__ == "__main__":
    
    load_directory = "V:\\Uni 3\\Project\\Frames\\"
    save_directory = "V:\\Uni 3\\Project\\OpticalFlow\\"
    threads=[]

    total_folders = len(os.listdir(load_directory))
    current_folder = 0

    for scene in os.listdir(load_directory):
        #print("Processing",scene,"folder")
        
        scenes_directory = os.path.join(load_directory, scene)
        save_folder = os.path.join(save_directory, scene)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        else:
            flow_folder = os.path.join(save_folder, "PointsFlow")
            if not os.path.exists(flow_folder): 
                os.mkdir(flow_folder)
                for subscene in os.listdir(scenes_directory):
                    frame_dir = os.path.join(scenes_directory,subscene)
                    #print("Processing",subscene)
                   
                    while len(threads) >= 12:
                        for thread in threads:
                            if not thread.is_alive():
                                threads.remove(thread)
                    
                    p1 = multiprocessing.Process(target=process_clip, args=(frame_dir,flow_folder, subscene))
                    threads.append(p1)
                    p1.start() 
                        
            else:
                print("Contains PointsFlow.npy file already, skipping" ,scene)


        current_folder +=1 
        if current_folder % round(total_folders/100) == 0:
            print("Completed {} out of {} folders".format(current_folder, total_folders))
    
import pickle
import cv2
import numpy as np
import os
import time

start=time.time()



"V:\\Uni4\\SoloProject\Frames\\1333_500_5_3_10_C_False\\1-1\\C_1-1-1"
if True: #lazy indent
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
    list_dir = os.listdir("V:\\Uni4\\SoloProject\Frames\\1333_500_5_3_10_C_False\\1-1\\C_1-1-1")

    while(index < len(list_dir)-1):
        frame = cv2.imread(os.path.join("V:\\Uni4\\SoloProject\Frames\\1333_500_5_3_10_C_False\\1-1\\C_1-1-1",list_dir[index]))
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
                if (index == len(list_dir)-2): #last frame
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

    '''
    dif_tracks = []
    for track in tracks:
        dif_track = []
        for index in range(len(track)-1):
            dx, dy = track[index+1][0]-track[index][0], track[index+1][1]-track[index][1]
            dif_track.append([dx, dy])
        dif_tracks.append(dif_track)
    '''

    # Dense tracks:
    frame = cv2.imread(os.path.join("V:\\Uni4\\SoloProject\Frames\\1333_500_5_3_10_C_False\\1-1\\C_1-1-1",list_dir[index]))
    frames = len(list_dir)
    height, width, _ = frame.shape 
    print(height, width, frames)


    framewise_tracks = [[[None for k in range(height)] for j in range(width)] for i in range(frames)] # [index][x][y]
    print(len(framewise_tracks))
    print(len(framewise_tracks[0]))
    print(len(framewise_tracks[0][0]))

    for track in complete_tracks:
        
        index = track[0]

        for track_index in range(1, len(track)-1):
            
            x = track[track_index][0]
            y = track[track_index][1]

            dx, dy = track[track_index+1][0]-x, track[track_index+1][1]-y
            
            framewise_tracks[index+track_index-1][int(x)][int(y)] = [dx, dy]

    # TODO: How do I test this is working correctly?

    print(time.time()-start)
    '''
    
    with open("test file", 'wb') as fp: 
        pickle.dump(dif_tracks, fp)
    '''
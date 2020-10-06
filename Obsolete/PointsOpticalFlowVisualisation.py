import cv2
import numpy as np
import os
import multiprocessing

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def process_clip(frame_dir, flow_folder, subscene):

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


    track_len = 100
    detect_interval = 1
    tracks = []


    while(index < len(os.listdir(frame_dir))-1):
        frame = cv2.imread(os.path.join(frame_dir, os.listdir(frame_dir)[index]))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                        del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            draw_str(vis, (20, 20), 'track count: %d' % len(tracks))

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
        cv2.imshow('lk_track', vis)

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    load_directory = "V:\\Uni3\\Project\\Frames\\"
    save_directory = "V:\\Uni3\\Project\\OpticalFlow\\"
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
                   
                while len(threads) >= 1:
                    for thread in threads:
                        if not thread.is_alive():
                            threads.remove(thread)
                    
                p1 = multiprocessing.Process(target=process_clip, args=(frame_dir,flow_folder, subscene))
                threads.append(p1)
                p1.start() 



        current_folder +=1 
        if current_folder % round(total_folders/100) == 0:
            print("Completed {} out of {} folders".format(current_folder, total_folders))
    
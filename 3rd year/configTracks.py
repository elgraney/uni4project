import os
# This file runs the pipeline for the tracks optical flow experiment

#Config Tracks is as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; tracksflow, save to 'Frames' folder, do not replace existing
#   3. Data set creation; tracksflow
#   4. Estimation; tracksflow, save folder
#   5. Performance tracksflow; load folder

directory = os.path.split(os.path.abspath(os.curdir))[0]

path = "python preprocessing.py 4/3 500 5 3 10 single False"
os.system(path)
path = "python tracksOpticalFlow.py tracks Frames False"
os.system(path)
path = "python tracksEVD.py tracks"
os.system(path)
path = "python estimationV2.py tracks tracksPoints"
os.system(path)
path = "python featurePerformanceStats.py tracksPoints"
os.system(path)
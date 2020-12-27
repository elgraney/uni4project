import os

# This file runs multiple run-throughs of the pipeline with different preprocessing parameters

# pipeline stuctures are as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; denseflow
#   3. Data set creation; denseflow, absolute values, debuggging = false
#   4. Estimation; denseflow, save folder
#   5. Performance analysis; load folder

def default(id):
    '''
    run default pipeline stucture for points optical flow with relative features
    '''
    path = "python FrameOpticalFlow.py Points Frames True"
    os.system(path)
    path = "python evaluateDataSetsV2.py points True False"
    os.system(path)
    path = "python estimation.py points RP_{}".format(id)
    os.system(path)
    path = "python featurePerformanceStats.py RP_{}".format(id)
    os.system(path)

directory = os.path.split(os.path.abspath(os.curdir))[0]

path = "python preprocessing.py 4/3 500 5 3 10 single true"
os.system(path)
default("Default")

path = "python preprocessing.py 4/3 800 5 3 10 single true"
os.system(path)
default("HighWidth")

path = "python preprocessing.py 4/3 300 5 3 10 single true"
os.system(path)
default("LowWidth")

path = "python preprocessing.py 4/3 500 5 3 15 single true"
os.system(path)
default("HighFPS")

path = "python preprocessing.py 4/3 500 5 3 5 single true"
os.system(path)
default("LowFPS")

path = "python preprocessing.py 4/3 500 10 10 10 single true"
os.system(path)
default("LongClipOnly")

path = "python preprocessing.py 4/3 500 2 2 10 single true"
os.system(path)
default("ShortClip")

path = "python preprocessing.py 4/3 500 5 3 10 full"
os.system(path)
default("corners")


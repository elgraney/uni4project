import os

# This file is a direct follow on from preprepTestingScript.py
# It runs multiple run-throughs of the pipeline changing specifically the clip length and framerate

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

#test fps
path = "python preprocessing.py 4/3 500 5 3 9 single true"
os.system(path)
default("FPS9")

path = "python preprocessing.py 4/3 500 5 3 8 single true"
os.system(path)
default("FPS8")

path = "python preprocessing.py 4/3 500 5 3 7 single true"
os.system(path)
default("FPS7")

path = "python preprocessing.py 4/3 500 5 3 6 single true"
os.system(path)
default("FPS6")

path = "python preprocessing.py 4/3 500 5 3 5 single true"
os.system(path)
default("FPS5")

path = "python preprocessing.py 4/3 500 5 3 4 single true"
os.system(path)
default("FPS4")

path = "python preprocessing.py 4/3 500 5 3 3 single true"
os.system(path)
default("FPS3")

path = "python preprocessing.py 4/3 500 5 3 2 single true"
os.system(path)
default("FPS2")

#test clip length (using no remainder)
path = "python preprocessing.py 4/3 500 6 6 10 single true"
os.system(path)
default("cliplength6")

path = "python preprocessing.py 4/3 500 7 7 10 single true"
os.system(path)
default("cliplength7")

path = "python preprocessing.py 4/3 500 8 8 10 single true"
os.system(path)
default("cliplength8")

path = "python preprocessing.py 4/3 500 9 9 10 single true"
os.system(path)
default("cliplength9")

path = "python preprocessing.py 4/3 500 10 10 10 single true"
os.system(path)
default("cliplength10")
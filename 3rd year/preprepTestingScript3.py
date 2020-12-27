import os

# This file is a direct follow on from preprepTestingScript2.py
# It tests the combination of the best parameters from previous experiments to see if they work well together

# pipeline stuctures are as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; denseflow
#   3. Data set creation; denseflow, absolute values, debuggging = false
#   4. Estimation; denseflow, save folder
#   5. Performance analysis; load folder

def default(id):
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
path = "python preprocessing.py 4/3 500 6 6 8 single true"
os.system(path)
default("FPS9")


import os
# This file runs the pipeline for dense flow with absolute features

#Config 4 is as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; pointsflow, save to 'Frames' folder, do not replace existing
#   3. Data set creation; pointsflow, relative values, debuggging
#   4. Estimation; pointsflow, save folder
#   5. Performance analysis; load folder

directory = os.path.split(os.path.abspath(os.curdir))[0]

path = "python preprocessing.py 4/3 500 5 3 10 single False"
os.system(path)
path = "python FrameOpticalFlow.py Points Frames False"
os.system(path)
path = "python evaluateDataSetsV3.py points True"
os.system(path)
path = "python estimation.py points relPoints"
os.system(path)
path = "python featurePerformanceStats.py relPoints"
os.system(path)
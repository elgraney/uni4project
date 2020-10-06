import os
# This file runs the absolute features vs relative features experiment

# The pipeline is as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses
#   2. Framewise Optical Flow; denseflow, save to 'Frames' folder
#   3. Data set creation; denseflow, absolute values
#   4. Estimation; denseflow, save folder
#   5. Performance analysis; load folder

directory = os.path.split(os.path.abspath(os.curdir))[0]

path = "python preprocessing.py 4/3 500 5 3 8 single False"
os.system(path)
path = "python FrameOpticalFlow.py points Frames False"
os.system(path)
path = "python evaluateDataSetsV2.py points False "
os.system(path)
path = "python estimation.py points absPoints"
os.system(path)
path = "python featurePerformanceStats.py absPoints"
os.system(path)


path = "python preprocessing.py 4/3 500 5 3 8 single False"
os.system(path)
path = "python FrameOpticalFlow.py points Frames False"
os.system(path)
path = "python evaluateDataSetsV2.py points True"
os.system(path)
path = "python estimation.py points relPoints"
os.system(path)
path = "python featurePerformanceStats.py relPoints"
os.system(path)
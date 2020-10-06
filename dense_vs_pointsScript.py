import os
# This file runs the dense flow vs points flow experiment

directory = os.path.split(os.path.abspath(os.curdir))[0]

#Dense is as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; denseflow, save to 'Frames' folder, do not replace existing
#   3. Data set creation; denseflow, absolute values
#   4. Estimation; denseflow, save folder
#   5. Performance analysis; load folder
path = "python preprocessing.py 4/3 500 5 3 8 single False"
os.system(path)
path = "python FrameOpticalFlow.py Dense Frames False"
os.system(path)
path = "python evaluateDataSetsV2.py Dense False"
os.system(path)
path = "python estimation.py Dense EVD2Dense"
os.system(path)
path = "python featurePerformanceStats.py EVD2Dense"
os.system(path)

#Points is as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; pointsflow, save to 'Frames' folder, do not replace existing
#   3. Data set creation; pointsflow, absolute values
#   4. Estimation; pointsflow, save folder
#   5. Performance analysis; load folder
path = "python preprocessing.py 4/3 500 5 3 8 single False"
os.system(path)
path = "python FrameOpticalFlow.py Points Frames False"
os.system(path)
path = "python evaluateDataSetsV2.py Points False"
os.system(path)
path = "python estimation.py Points EVD2Points"
os.system(path)
path = "python featurePerformanceStats.py EVD2Points"
os.system(path)
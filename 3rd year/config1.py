import os
# This file runs the pipeline for dense flow with absolute features

#Config 1 is as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; denseflow, save to 'Frames' folder, do not replace existing
#   3. Data set creation; denseflow, absolute values, debuggging = false
#   4. Estimation; denseflow, save folder
#   5. Performance analysis; load folder

directory = os.path.split(os.path.abspath(os.curdir))[0]

path = "python preprocessing.py 4/3 500 5 3 10 single False"
os.system(path)
path = "python FrameOpticalFlow.py dense Frames False"
os.system(path)
path = "python evaluateDataSetsV3.py dense False False"
os.system(path)
path = "python estimation.py dense absDense"
os.system(path)
path = "python featurePerformanceStats.py absDense"
os.system(path)
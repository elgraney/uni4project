import os
#Config 5 is as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses
#   2. Framewise Optical Flow; pointsflow, save to 'Frames' folder
#   3. Data set creation; pointsflow, absolute values, debuggging = false
#   4. Estimation; pointsflow, save folder
#   5. Performance pointsflow; load folder

directory = os.path.split(os.path.abspath(os.curdir))[0]

path = "python preprocessing.py 4/3 500 5 3 10 single False"
os.system(path)
path = "python FrameOpticalFlow.py points Frames"
os.system(path)
path = "python directionalFramesEVD.py points False False"
os.system(path)
path = "python estimationDirectional.py points directionalAbsPoints"
os.system(path)
path = "python featurePerformanceStats.py directionalAbsPoints"
os.system(path)
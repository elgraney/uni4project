import os
# This file is a variation on preprepTestingScript2.py
# It runs simlar tests except with tracks optical flow rather than points

# pipeline stuctures are as follows:
#   1. Preprocessing; ratio, width, interval time, remainder time, FPS, focuses, do not replace existing
#   2. Framewise Optical Flow; denseflow
#   3. Data set creation; denseflow, absolute values, debuggging = false
#   4. Estimation; denseflow, save folder
#   5. Performance analysis; load folder

def default(id):
    path = "python tracksOpticalFlow.py tracks Frames True"
    os.system(path)
    path = "python tracksEVD.py tracks"
    os.system(path)
    path = "python estimation.py tracks TP_{}".format(id)
    os.system(path)
    path = "python featurePerformanceStats.py TP_{}".format(id)
    os.system(path)

directory = os.path.split(os.path.abspath(os.curdir))[0]
#path = "python preprocessing.py 4/3 500 5 3 10 single true"
#os.system(path)
#default("Default")

path = "python preprocessing.py 4/3 500 5 3 9 single true"
os.system(path)
default("FPS9")

path = "python preprocessing.py 4/3 500 5 3 8 single true"
os.system(path)
default("FPS8")

path = "python preprocessing.py 4/3 500 5 3 7 single true"
os.system(path)
default("FPS7")

path = "python preprocessing.py 4/3 500 5 3 5 single true"
os.system(path)
default("FPS5")

path = "python preprocessing.py 4/3 500 5 3 3 single true"
os.system(path)
default("FPS3")

path = "python preprocessing.py 4/3 500 5 3 2 single true"
os.system(path)
default("FPS2")


#test clip length (using no remainder)

path = "python preprocessing.py 4/3 500 2 2 10 single true"
os.system(path)
default("cliplength2-2")

path = "python preprocessing.py 4/3 500 3 3 10 single true"
os.system(path)
default("cliplength3-3")

path = "python preprocessing.py 4/3 500 4 4 10 single true"
os.system(path)
default("cliplength4-4")

path = "python preprocessing.py 4/3 500 6 6 10 single true"
os.system(path)
default("cliplength6-6")

path = "python preprocessing.py 4/3 500 7 7 10 single true"
os.system(path)
default("cliplength7-7")
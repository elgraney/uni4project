import os
import time

directory = os.path.split(os.path.abspath(os.curdir))[0]
filename = "Test"

replace = False
    
ratio = "4/3"
width = "500"
interval = "5"
remainder = "3"
frame_rate = "3"
focus = "C"
max_loops = "1" # only 1 clip per video for speeeeed

maxCorners = 500
qualityLevel = 0.01
minDistance = 5
blockSize = 10
winSize = 25
maxLevel = 3

kernel = "rbf"
gamma = "auto"
C = 0.001


preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))

start = time.time()

os.system("python preprocessing.py "+preprocessing_code+" "+str(replace)) #using defaults
print("Duration:")
print(time.time()-start)
os.system("python 2dOpticalFlow.py "+preprocessing_code+" "+opflow_code+" "+str(replace)) #using defaults
print("Duration:")
print(time.time()-start)
os.system("python 2dFeatureSelection.py "+preprocessing_code+" "+opflow_code+" "+filename) #using defaults
print("Duration:")
print(time.time()-start)
os.system("python SVM.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+svm_code) #using defaults

print("Duration:")
print(time.time()-start)
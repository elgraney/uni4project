import os
import time

directory = os.path.split(os.path.abspath(os.curdir))[0]

start = time.time()

os.system("python preprocessing.py") #using defaults
print("Duration:")
print(time.time()-start)
os.system("python 2dOpticalFlow.py") #using defaults
print("Duration:")
print(time.time()-start)
os.system("python 2dFeatureSelection.py") #using defaults
print("Duration:")
print(time.time()-start)
os.system("python SVM.py") #using defaults

print("Duration:")
print(time.time()-start)
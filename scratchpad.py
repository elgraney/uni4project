import pickle
import cv2
import numpy as np
import os
import time
import evaluation

load_dir = "V:\\Uni4\\SoloProject\\DataSets\\16_9_300_10_3_3_C_1\\500_0.001_10_10_50_3\\test"
data = pickle.load( open( load_dir, "rb") )
print(data)
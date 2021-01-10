import pickle
import cv2
import numpy as np
import os
import time
import evaluation

start=time.time()

evaluation.test_ranking("V:\\Uni4\\SoloProject\\Outputs\\4_3_500_5_3_10_C_False_500_0.001_10_10_25_3\\tests")

print(time.time()-start)

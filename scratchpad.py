import pickle
import cv2
import numpy as np
import os
import time
import evaluation

start=time.time()

evaluation.test_ranking("V:\\Uni4\\SoloProject\\Outputs\\Unique_Code")

print(time.time()-start)

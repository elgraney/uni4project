from sklearn import preprocessing
import constants
import numpy as np

def standard_scale(data):
    for key, value in data.items():
        if key != constants.feature0:
            scaler = preprocessing.StandardScaler().fit(np.array(value).reshape(-1,1))
            data[key] = list((scaler.transform(np.array(value).reshape(-1,1))).flatten())
    return data
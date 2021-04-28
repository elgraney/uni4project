import os
import pickle
import numpy as np
import machineLearning as ML
from sklearn.linear_model import LogisticRegression




if __name__ == '__main__':
    with open("Best_clf", "rb") as file_in:
        clf = pickle.load(file_in)[0]

    preprocessing_code, opflow_code, filename = "16_9_500_5_3_5_C_1", "500_0.0001_5_15_25_3", "Test"
    ml_code = "Logistic_Regression"

    load_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Datasets", preprocessing_code, opflow_code, filename)
    data = pickle.load( open( load_dir, "rb") )

    special_item_index = data["category"].index("10")

    special_item_data = np.array([data[feature][special_item_index] for feature in data.keys() if feature != "category"])
    
    print("Special video feature values:")
    print(special_item_data)

    prediction = clf.predict(special_item_data.reshape(1, -1))
    print("\npredicted wind Force:", prediction)


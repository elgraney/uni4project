import os
import evaluation
import os
import evaluation


def runthrough(preprocessing_code, opflow_code, filename, replace, svm_code):
    path = "python preprocessingHome.py "+preprocessing_code+" "+str(replace)
    os.system(path)
    path = "python 2dOpticalFlow.py "+preprocessing_code+" "+opflow_code+" "+str(replace)
    os.system(path)
    path = "python 2dFeatureSelection.py "+preprocessing_code+" "+opflow_code+" "+filename
    os.system(path)
    path = "python linearRegression.py "+preprocessing_code+" "+opflow_code+" "+filename+" Logistic_Regression"
    print(path)
    os.system(path)


if __name__ == "__main__":
    replace = False
    preprocessing_code, opflow_code, filename = "16_9_500_5_3_5_C_1", "500_0.0001_5_15_25_3", "Test"

    runthrough(preprocessing_code, opflow_code, filename, replace, "Logistic_Regression")


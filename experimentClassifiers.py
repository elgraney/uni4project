import os
import evaluation
'''
Experiment 1
    Uses general data, not excluding camera motion

    1. Preproccesing:       ratio1_ratio2, width, interval, remainder, frame_rate, focus, replace
    2. 2d OpticalFlow:      preprocessing code, opflow_code = (replace, maxCorners, qualityLevel, minDistance), replace
    3. 2d FeatureSelection: preprocessing_code = "4_3_500_5_3_10_C_False"
                            opflow_code = "500_0.001_10_10_25_3"
                            filename = "default"
    4. SVM:                 " "

'''

def runthrough(preprocessing_code, opflow_code, filename, replace, svm_code):
    path = "python preprocessing.py "+preprocessing_code+" "+str(replace)
    os.system(path)
    path = "python 2dOpticalFlow.py "+preprocessing_code+" "+opflow_code+" "+str(replace)
    os.system(path)
    path = "python 2dFeatureSelection.py "+preprocessing_code+" "+opflow_code+" "+filename
    os.system(path)
    path = "python linearRegression.py "+preprocessing_code+" "+opflow_code+" "+filename+" Logistic_Regression"
    print(path)
    os.system(path)



if __name__ == "__main__":
    directory = os.path.split(os.path.abspath(os.curdir))[0]

    filename = "Test"

    replace = False
    
    ratio = "4/3"
    width = "600"
    interval = "5"
    remainder = "3"
    frame_rate = "5"
    focus = "C"
    max_loops = "1" # only 1 clip per video for speeeeed

    maxCorners = 500
    qualityLevel = 0.0001
    minDistance = 5
    blockSize = 15
    winSize = 25
    maxLevel = 3



    preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
    opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
    ml_code = "Logistic_Regression"

    runthrough(preprocessing_code, opflow_code, filename, replace, ml_code)

    for kernel in ["rbf", "linear", "poly"]:
        for gamma in ["auto", "scale"]:
            for C in ["0.001","0.01","0.1","1", "10", "100"]:
                svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))
                path = "python SVM.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+svm_code
                print(path)
                os.system(path)

    max_depth = "rbf"
    min_samples_split = "auto"
    min_samples_leaf = 1

    for max_depth in [None, 15, 20, 25, 30]:
        for min_samples_split in [2,3,5,10,20, 25, 30]:
            for min_samples_leaf in [1,2,5,10, 20]:
                ml_code = "{}_{}_{}".format(str(max_depth), str(min_samples_split), str(min_samples_leaf))
                path = "python decisionTree.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+ml_code
                print(path)
                os.system(path)

    for alpha in [1]:
        for n in  [120, 135, 150,165, 185, 200, 220, 250, 300]:
            ml_code = "{}_{}".format(str(alpha), str(n))
            path = "python MLP.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+ml_code
            print(path)
            os.system(path)
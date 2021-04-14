import os
import evaluation
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
    directory = os.path.split(os.path.abspath(os.curdir))[0]
    filename = "Test"

    replace = False
    
    ratio = "16/9"
    width = "500"
    interval = "29"
    remainder = "3"
    frame_rate = "10"
    focus = "C"
    max_loops = "1" # only 1 clip per video for speeeeed

    maxCorners = 500
    qualityLevel = 0.0001
    minDistance = 5
    blockSize = 15
    winSize = 25
    maxLevel = 3

    # svm vars
    kernel = "rbf"
    gamma = "scale"
    C = 1.0

    #DT vars
    max_depth = 25
    min_samples_split = 20
    min_samples_leaf = 3

    #MLP vars
    alpha = 1
    n = 175

    
    preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
    opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
    svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))
    DT_code = "{}_{}_{}".format(str(max_depth), str(min_samples_split), str(min_samples_leaf))
    MLP_code = "{}_{}".format(str(alpha), str(n))

    runthrough(preprocessing_code, opflow_code, filename, replace, "Logistic_Regression")

    for interval in [5, 10, 20, 29]:
        for frame_rate in [5,8, 10]:
                preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
                ml_code = "Logistic_Regression"
                runthrough(preprocessing_code, opflow_code, filename, replace, ml_code)
                path = "python SVM.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+svm_code
                print(path)
                os.system(path)

                path = "python decisionTree.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+DT_code
                print(path)
                os.system(path)

                path = "python MLP.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+MLP_code
                print(path)
                os.system(path)



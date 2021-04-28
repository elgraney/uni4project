import os
import evaluation


def runthrough(preprocessing_code, opflow_code, filename, replace, svm_code):
    path = "python preprocessing.py "+preprocessing_code+" "+str(replace)
    os.system(path)
    path = "python 2dOpticalFlow.py "+preprocessing_code+" "+opflow_code+" "+str(replace)
    os.system(path)
    path = "python 2dFeatureSelection.py "+preprocessing_code+" "+opflow_code+" "+filename
    os.system(path)
    path = "python SVM.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+svm_code
    print(path)
    os.system(path)


if __name__ == "__main__":
    replace = False

    directory = os.path.split(os.path.abspath(os.curdir))[0]
    filename = "Test"

    
    ratio = "4/3"
    width = "300"
    interval = "3"
    frame_rate = "5"


    maxCorners = 500
    qualityLevel = 0.001
    minDistance = 10
    blockSize = 10
    winSize = 50
    maxLevel = 3

    kernel = "rbf"
    gamma = "auto"
    C = 1


    preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(1),str(frame_rate),str("".join("C")), str(1))
    opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
    svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))

    runthrough(preprocessing_code, opflow_code, filename, replace, svm_code)


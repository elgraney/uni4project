import os
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
    path = "python SVM.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+svm_code
    print(path)
    os.system(path)




if __name__ == "__main__":
    directory = os.path.split(os.path.abspath(os.curdir))[0]
    filename = "Test"

    replace = False
    
    ratio = "4/3"
    width = "500"
    interval = "5"
    remainder = "3"
    frame_rate = "10"
    focus = "C"
    max_loops = "1" # only 1 clip per video for speeeeed
    

    


    maxCorners = 500
    qualityLevel = 0.001
    minDistance = 10
    blockSize = 10
    winSize = 25
    maxLevel = 3


    kernel = "rbf"
    gamma = "auto"
    C = 1
    

    for width in ["300", "500", "700" ]:
        for interval in ["1", "2", "3", "8"]:
            for frame_rate in ["1", "2", "3", "4"]:
                for maxCorners in ["500", "1000", "2000"]:
                    for minDistance in ["5", "10", "20", "50"]:
                        for qualityLevel in ["0.01", "0.001", "0.0001"]:
                            for blockSize in ["10"]:
                                for winSize in ["25"]:
                                    for maxLevel in ["3"]:

                                        preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
                                        opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
                                        svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))
                                        runthrough(preprocessing_code, opflow_code, filename, replace, svm_code)
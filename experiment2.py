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
    path = "python SVM.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+svm_code
    print(path)
    os.system(path)


def preprocessing_experiment(opflow_flow, svm_code):
    for width in ["300", "500", "700" ]:
        for interval in ["2", "3", "8"]:
            for frame_rate in ["2", "3", "4"]:
                preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
                runthrough(preprocessing_code, opflow_code, filename, replace, svm_code)



def SVM_experiment(preprocessing_code, opflow_flow):
    for kernel in ["rbf"]:
        for gamma in ["auto","0.0001","0.001", "0.01","0.1","1", "10", "100", "1000"]:
            for C in ["0.001","0.01","0.1","1", "10", "100"]:
                svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))
                runthrough(preprocessing_code, opflow_code, filename, replace, svm_code)



def best_opflow():
    directory = "V:\\Uni4\\SoloProject\\Outputs\\"
    best = {}
    for preprocessing_test in os.listdir(directory):
        preprep_dir = os.path.join(directory, preprocessing_test)
        
        for opflow_test in os.listdir(preprep_dir):
            opflow_dir = os.path.join(directory, opflow_test)

            for SVM_test in os.listdir(opflow_dir):
                svm_dir = os.path.join(directory, SVM_test)
                for test in os.listdir(svm_dir):
                    _, lenient_stats = evaluation.test_ranking(os.path.join(svm_dir, test), False)
                    lenient_best = lenient_stats[-1]
                    try:
                        if lenient_best > best[opflow_test]:
                            best[opflow_test] = lenient_best
                    except KeyError:
                        best[best_opflow] = lenient_best
                        
    print(best)

    return [1]



if __name__ == "__main__":
    directory = os.path.split(os.path.abspath(os.curdir))[0]
    filename = "Test"

    replace = False
    
    ratio = "4/3"
    width = "300"
    interval = "3"
    remainder = "3"
    frame_rate = "3"
    focus = "C"
    max_loops = "1" # only 1 clip per video for speeeeed

    maxCorners = 500
    qualityLevel = 0.01
    minDistance = 5
    blockSize = 10
    winSize = 25
    maxLevel = 3

    kernel = "rbf"
    gamma = "auto"
    C = 1


    preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
    opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
    svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))
    

    preprocessing_experiment(opflow_code, svm_code)
    preprep_best = [1] #best_preprep()
    for preprocessing_code in preprep_best:
        #for top 5 in preprocessing_flow_exp
        SVM_experiment(preprocessing_code, opflow_code)
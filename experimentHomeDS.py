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
    path = "python SVM.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+svm_code
    print(path)
    os.system(path)


def preprocessing_experiment(opflow_flow, svm_code):
    for width in ["300", "400", "500"]:
        for interval in ["10","29" ]:
            for frame_rate in ["3", "5", "10"]:
                preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
                runthrough(preprocessing_code, opflow_code, filename, replace, svm_code)


def optical_flow_experiment(preprocessing_code, svm_code):
    for maxCorners in ["500"]:
        for minDistance in ["5"]:
            for qualityLevel in ["0.001"]:
                for blockSize in ["10"]:
                    for winSize in ["25"]:
                        for maxLevel in ["3"]:
                            opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
                            runthrough(preprocessing_code, opflow_code, filename, replace, svm_code)

def SVM_experiment(preprocessing_code, opflow_flow):
    for kernel in ["rbf"]:
        for gamma in ["auto", "scale"]:
            for C in ["0.001","0.01","0.1","1", "10", "100"]:
                svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))
                runthrough(preprocessing_code, opflow_code, filename, replace, svm_code)



def best_preprep(n = 10):
    directory = "V:\\Uni4\\SoloProject\\Outputs\\"
    best = {}
    for preprocessing_test in os.listdir(directory):
        preprep_dir = os.path.join(directory, preprocessing_test)
        
        for opflow_test in os.listdir(preprep_dir):
            opflow_dir = os.path.join(preprep_dir, opflow_test)

            for SVM_test in os.listdir(opflow_dir):
                svm_dir = os.path.join(opflow_dir, SVM_test)
                _, lenient_stats = evaluation.test_ranking(svm_dir, False)
                lenient_best = list(lenient_stats.values())[-1]
                try:
                    if lenient_best > best[preprocessing_test]:
                        best[preprocessing_test] = lenient_best
                except KeyError:
                    best[preprocessing_test] = lenient_best  
    print(list({k: v for k, v in sorted(best.items(), key=lambda item: item[1], reverse=True)}.values())[:n])
    return_best = list({k: v for k, v in sorted(best.items(), key=lambda item: item[1], reverse=True)}.keys())[:n]
    return return_best


def best_opflow(n = 10):
    directory = "V:\\Uni4\\SoloProject\\Outputs\\"
    best = {}
    for preprocessing_test in os.listdir(directory):
        preprep_dir = os.path.join(directory, preprocessing_test)
        
        for opflow_test in os.listdir(preprep_dir):
            opflow_dir = os.path.join(preprep_dir, opflow_test)

            for SVM_test in os.listdir(opflow_dir):
                svm_dir = os.path.join(opflow_dir, SVM_test)
                _, lenient_stats = evaluation.test_ranking(svm_dir, False)
                lenient_best = list(lenient_stats.values())[-1]
                try:
                    if lenient_best > best[opflow_test]:
                        best[opflow_test] = lenient_best
                except KeyError:
                    best[opflow_test] = lenient_best  
    print(list({k: v for k, v in sorted(best.items(), key=lambda item: item[1], reverse=True)}.values())[:n])
    return_best = list({k: v for k, v in sorted(best.items(), key=lambda item: item[1], reverse=True)}.keys())[:n]
    return return_best



if __name__ == "__main__":
    directory = os.path.split(os.path.abspath(os.curdir))[0]
    filename = "Test"

    replace = False
    
    ratio = "16/9"
    width = "500"
    interval = "30"
    remainder = "3"
    frame_rate = "10"
    focus = "C"
    max_loops = "1" # only 1 clip per video for speeeeed

    maxCorners = 500
    qualityLevel = 0.001
    minDistance = 10
    blockSize = 10
    winSize = 50
    maxLevel = 3

    kernel = "rbf"
    gamma = "auto"
    C = 1


    preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
    opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
    svm_code = svm_code = "{}_{}_{}".format(str(kernel), str(gamma), str(C))
    

    preprocessing_experiment(opflow_code, svm_code)



    optical_flow_experiment(preprocessing_code, svm_code)

    SVM_experiment(preprocessing_code, opflow_code)


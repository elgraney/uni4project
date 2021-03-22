import os
import evaluation
'''
for MLP with best settings for preprep

'''

def runthrough(preprocessing_code, opflow_code, filename, replace, ml_code):
    path = "python preprocessing.py "+preprocessing_code+" "+str(replace)
    os.system(path)
    path = "python 2dOpticalFlow.py "+preprocessing_code+" "+opflow_code+" "+str(replace)
    os.system(path)
    path = "python 2dFeatureSelection.py "+preprocessing_code+" "+opflow_code+" "+filename
    os.system(path)
    path = "python MLP.py "+preprocessing_code+" "+opflow_code+" "+filename+" "+ml_code
    print(path)
    os.system(path)



def best_preprep(n = 10):
    directory = "X:\\uni4\\Solo project 2\\Dataset 3 complete outputs\\"
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
    directory = "X:\\uni4\\Solo project 2\\Dataset 3 complete outputs\\"
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
    
    ratio = "4/3"
    width = "500"
    interval = "5"
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

    #mlp params....


    preprocessing_code = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops))
    opflow_code = "{}_{}_{}_{}_{}_{}".format(str(maxCorners), str(qualityLevel), str(minDistance), str(blockSize), str(winSize),str(maxLevel))
    ml_code = "{}_{}_{}".format(str("rbf"), str(1), str(1)) # TODO
    
    # Get 10 best of each code
    # Mix them
    # Return best results

    preprep_best = best_preprep(3)
    opflow_best = best_opflow(3)
    for preprocessing_code in preprep_best:
        print(preprocessing_code)
        for opflow_code in opflow_best:
            print(opflow_code)

            runthrough(preprocessing_code, opflow_code, filename, replace, ml_code)

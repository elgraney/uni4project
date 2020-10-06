import os
import sys

# This file calculates and saves the average accuracy for each feature in a test using data from estimation log files

if __name__ == "__main__":

    save_dir = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Outputs") 

    if len(sys.argv)>1:
        try:
            save_dir = os.path.join(save_dir, sys.argv[1])
            if not os.path.exists(save_dir):
               raise Exception("Bad argument; argument 1")
        except IOError:
            print(sys.argv[1])
            raise Exception("Bad argument; argument 1")

    print(os.path.join(save_dir, "Best.txt"))
    with open(os.path.join(save_dir, "Best.txt"),"r") as result_file:
        lines = result_file.readlines()
    result_file.close()

    new_lines = []
    percs = []
    lines = lines[1:]
    for line in lines:
        new_line = line.split("Test ")[1]
        new_line = new_line.split(":")[0]
        new_line = new_line.split(", ")
        new_lines.append(new_line)
        
        exact_percentage = line.split("Exact Accuracy=")[1]
        exact_percentage = float(exact_percentage.split(",")[0])
        percs.append(exact_percentage)
 
 
    rank = 1
    score_list = [0] * len(new_lines[0])
    percs_list = [0] * len(new_lines[0])
    for line, perc in zip(new_lines, percs):
        for item_index in range(len(line)):
            if line[item_index] == "True":
                percs_list[item_index] += perc
                score_list[item_index] += rank
            else:
                pass
        rank += 1
    
    score_list[:] = [x / 1000 for x in score_list]

    av_percs_list = []
    for percentage in percs_list:
        percentage = percentage / (len(lines)/2)
        av_percs_list.append(round(percentage, 3))

    with open(os.path.join(save_dir, "Stats.txt"),"w") as result_file:
        result_file.write(str(score_list))
        result_file.write("\n"+str(av_percs_list))
    result_file.close()

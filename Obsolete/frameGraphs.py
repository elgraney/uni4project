# take in all optical flow values
# map as graph
# present graph 

import numpy as np
import matplotlib.pyplot as plt
import collections
import statistics 
import os
import pickle
  

load_dir = "V:\\Uni3\\Project\\OpticalFlow\\"
save_dir = "V:\\Uni3\\Project\\Graphs\\"

for video in os.listdir(load_dir):
    save_folder = os.path.join(save_dir, video)
    load_folder = os.path.join(load_dir, video, "DenseFlow")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(os.path.join(save_folder, "frameScatter")) or not os.path.exists(os.path.join(save_folder, "frameMagnitudesPlot")):
        for file in os.listdir(load_folder):
            print("Processing ",file,"...")
            flow_list = pickle.load( open( os.path.join(load_folder, file), "rb" ))
    
            os.mkdir(os.path.join(save_folder, "frameScatter"))
            os.mkdir(os.path.join(save_folder, "frameMagnitudesPlot"))

            for frame_index in range(len(flow_list)):
                print("processing frame",str(frame_index+1))
                xlist = []
                ylist = [] 
                magnitudes = []
                for x in range(len((flow_list[0][0]))):
                    for y in range(len((flow_list[0]))):
                        xlist.append(flow_list[frame_index][y][x][0])
                        ylist.append(flow_list[frame_index][y][x][1])
                        magnitude = float(np.linalg.norm(flow_list[frame_index][y][x]))
                        if magnitude > 0.01:
                            magnitudes.append(round(magnitude,1))

                        
                        
                print("plotting scatter...")
                plt.scatter(xlist, ylist, s=0.5)
                plt.savefig(os.path.join(save_folder, "frameScatter\\"+video+str(frame_index+1)+".png"))
                plt.clf()


                counter = collections.Counter(magnitudes)
                od = collections.OrderedDict(sorted(counter.items()))
                print("plotting magnitudes...")


                plt.plot(list(od.keys()), list(od.values()))
                plt.savefig(os.path.join(save_folder, "frameMagnitudesPlot\\"+video+"-"+str(frame_index+1)+".png"))
                plt.clf()
                print("finished")


            print("Set to only examine 1 video per source for brevity")
            break

import os
import pickle
import time
import commonFunctions
import matplotlib.pyplot as plt
import sys


def plot_tracks2(load_dir):
    '''
    this time we plot x and y on the same axis.


    '''
    for scene in os.listdir(load_dir):
        print(scene)
        load_directory = os.path.join(load_dir, scene)
        tracks_list = []
        for folder_name in os.listdir(load_directory):
            try:
                tracks_list += pickle.load( open( os.path.join(load_directory, folder_name, "Tracks"), "rb" ))
            except:
                print("failed to load file {}".format(os.path.join(load_directory, folder_name)))
                return

            n = 50
            i = 0
            for track in tracks_list:
                if i % n == 0:
                    y = [dy for (dx, dy) in track[1:]]
                    x = [dx for (dx, dy) in track[1:]]
                    plt.plot(x, y)
                i+=1
            plt.savefig(os.path.join("V:\\Uni4\\SoloProject\\tracksPlots",scene + folder_name))
            plt.clf()



def plot_tracks(load_dir):
    for scene in os.listdir(load_dir):
        print(scene)
        load_directory = os.path.join(load_dir, scene)
        tracks_list = []
        for folder_name in os.listdir(load_directory):
            try:
                tracks_list += pickle.load( open( os.path.join(load_directory, folder_name, "Tracks"), "rb" ))
            except:
                print("failed to load file {}".format(os.path.join(load_directory, folder_name)))
                return

            n = 50
            i = 0
            for track in tracks_list:
                if i % n == 0:
                    y = [dy for (dx, dy) in track[1:]]
                    x = range(int(track[0][0]), int(track[0][0])+len(track[1:]))
                    plt.plot(x, y)
                i+=1
            plt.savefig(os.path.join("V:\\Uni4\\SoloProject\\tracksPlots",scene + folder_name))
            plt.clf()

        # to do this properly we need a few things
        '''
        1. Track indexes back in. DONE
        2. less points to track per image
        3. With camera motion correction and without
        '''

        


if __name__ == '__main__':
    start = time.time()

    preprocessing_code, opflow_code, filename = commonFunctions.code_inputs(sys.argv)

    opflow_code = "Flow Test 1"
    
    
    load_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "OpticalFlow", preprocessing_code, opflow_code)
    save_directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Datasets", preprocessing_code + "_" + opflow_code)

    plot_tracks2(load_directory)




    
    end=time.time()
    print("Moment of truth...")
    print("threading global averages time:")
    print(str(end - start))


    #CURRENT RUN TIME APPROX 10182 (10 threads)
    #Approx same with nopython mean. Try same with sd.
    #Reduced to 7543 with 10 threads and sd
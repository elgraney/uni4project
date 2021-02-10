import time
import pickle
import statistics
from matplotlib import pyplot as plt

if __name__ == '__main__':
    start = time.time()

    # handle command line args

    load_dir = "V:\\Uni4\\SoloProject\\DataSets\\4_3_500_5_3_10_C_False_500_0.001_10_10_25_3\\default"
    data = pickle.load( open( load_dir, "rb") )

    stats_by_force = {}

    for feature in data.keys():
        if feature != "category":
            stats_by_force[feature] = [[] for x in range(13)] # indices are the force

    for index in range(len(data[list(data.keys())[0]])):
 
        force = data["category"][index]
        for feature in data.keys():
            if feature != "category":
                try:
                    stats_by_force[feature][eval(force)].append(data[feature][index])
                except Exception:
                    continue

    for feature in stats_by_force.keys():
        for force in range(13):
            if stats_by_force[feature][force] != []:
                stats_by_force[feature][force] = statistics.mean(stats_by_force[feature][force])
            else: stats_by_force[feature][force] = 0

    for stat in stats_by_force.keys():
        print(stat)
        plt.plot(stats_by_force[stat])
        plt.show()

    end=time.time()
    print("estimation duration:")
    print(str(end - start))
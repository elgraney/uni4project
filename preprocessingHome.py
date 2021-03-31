import preprocessing as pp
import time
import os
import commonFunctions
import constants
import multiprocessing
import shutil


def create_dataset(raw_dir):
    raw_filenames = []
    new_filenames = []
    for folder in os.listdir(raw_dir):
        split_list = folder.split("--")
        if len(split_list) > 1:
            mph = split_list[-1].split(".")[0]
            beaufort = commonFunctions.mph_to_beaufort(eval(mph))

            raw_filenames.append(folder)
            new_filenames.append(pp.cleanid(split_list[0]) +"-"+str(beaufort))

    total = len(new_filenames)

    return total, raw_filenames, new_filenames


if __name__ == "__main__":
    start = time.time()

    ratio, width, interval, remainder, frame_rate, focus, max_loops, replace = pp.inputs()

    directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "home_camera")
    total, raw_filenames, new_filenames = create_dataset(directory)


    ratio_split = ratio.split("/")
    save_directory = os.path.join(os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Frames"), "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops)))
    ratio = eval(ratio)
    
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    count = 0

    threads = []
    for raw, new in zip(raw_filenames, new_filenames):
        current_path = os.path.join(directory,raw)

        save_path = os.path.join(save_directory, new)
                    
        # Setup folder for outputs
        if os.path.exists(save_path) and replace == True:
            shutil.rmtree(save_path)
        elif os.path.exists(save_path) and replace == False:
            #print("skipping", filename)
            continue
                    
        print("Processing", raw, "as", new)

                    
        os.mkdir(save_path)
                    
        for focus_point in focus:
            while len(threads) >= constants.THREADS:
                for thread in threads:
                    if not thread.is_alive():   
                        threads.remove(thread)
                        
            p1 = multiprocessing.Process(target=pp.process_video, args=(current_path, save_path, new, focus_point, ratio, width, interval, remainder, frame_rate, max_loops))
            threads.append(p1)
            p1.start()
                   
        count +=1 
        if count % 10 == 0:
            print("Completed {} out of {} folders".format(count, total))

    for thread in threads:
        thread.join()

    end=time.time()
    print("Completed preprocessing")
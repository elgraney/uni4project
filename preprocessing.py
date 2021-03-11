from moviepy.editor import *
from moviepy.video.fx.all import resize
from moviepy.video.fx.all import crop
import os
import multiprocessing
import time
import random
import string
import shutil
import constants

# This file handles the preparation of video files into useful data items

def ratioCrop(video, goalRatio, location):
    '''
    Reduces a video to the goal ratio, centered on a certain subsection of the frame
    video: the video being cropped 
    goalRation: the width:height ratio to reduce to
    location: the subection of the frame to focus on
    '''
    (width, height) = video.size
    ratio = width / float(height)
    if ratio > goalRatio:
        # crop the left and right edges:
        new_width = int(goalRatio * height)
        if "l" in location.lower():
            offset1 = 0
            offset2 = (width - new_width)
        elif "r" in location.lower():
            offset1 = (width - new_width)
            offset2 = 0
        else:
            offset1 = (width - new_width) / 2
            offset2 = offset1

        x1 = offset1
        x2 = width - offset2
        y1 = 0
        y2 = height
    else:
        # ... crop the top and bottom:
        new_height = int(width / goalRatio)
        if "t" in location.lower():
            offset1 = 0
            offset2 = (height - new_height)
        elif "b" in location.lower():
            offset1 = (height - new_height)
            offset2 = 0
        else:
            offset1 = (height - new_height) / 2
            offset2 = offset1
       
        x1 = 0
        x2 = width
        y1 = offset1
        y2 = height - offset2
          
    cropped_video = crop(video, x1, y1, x2, y2)

    return cropped_video


def reduce(video, goal): 
    '''
    Reduce the resolution of a video to the goal size
    '''
    return video.resize(width = goal)


def extract_frames(clip, times, imgdir):
    '''
    Save still frames from the video at regular time intervals
    '''
    for t in times:
        imgpath = os.path.join(imgdir, '{}.png'.format(t))
        clip.save_frame(imgpath, t)


def process_video(load_path, save_path, fileID, location, ratio, width, interval, remainder, frame_rate, max_loops = 100):
    '''
    load_path: the directory of the video to load
    save_path:
    fileID: the name of the file and the wind force
    location: the area of video to centre on.
    '''

    video = VideoFileClip(load_path).without_audio()
    cropped_video = ratioCrop(video, ratio, location)
    
    if cropped_video.size[0] > width:
        reduced_video = reduce(cropped_video, width)
    else: 
        reduced_video = cropped_video

    video_length = reduced_video.duration

    # Split the full video into multiple clips each 'interval' seconds long
    clips = []
    #setting an upper limit on the number of subclips that can be made
    loops = 0
    for start_time in range(0,int(video_length)-interval, interval):
        clips.append(reduced_video.subclip(start_time, start_time + interval))
        loops+=1
        if loops >=max_loops:
            #print(fileID," has exeeded maximum length, cutting at 6 subclips")
            break

    # If remaining video is longer than <remainder> seconds duration
    if video_length % interval >= remainder and loops < max_loops:
        clips.append(reduced_video.subclip((video_length // interval)*interval, video_length))
            
    #print("Writing",fileID,"...")
    if len(clips) >0:
        for index in range(len(clips)):
            # Store low-res, low-framrate subclip
            # Don't save video for space conservation
            #clips[index].write_videofile(video_path+"\\videos\\"+location+"_"+str(index+1)+"-"+fileID+".mp4",fps = frame_rate) 

            subclip_path = os.path.join(save_path, location+"_"+str(index+1)+"-"+str(fileID))
            os.mkdir(subclip_path)

            times = [x/frame_rate for x in range(0,int((clips[index].duration*frame_rate)+1))] #requires int; multiply by 10, +1 for maximised frames
            extract_frames(clips[index], times, subclip_path)
        #print("Finished",fileID,"...")
    else:
        os.rmdir(save_path)
    video.close()


def cleanid(filename):
     #Make sure the string is valid
    if len(filename.split("-")) > 2:
        parts = filename.split("-")
        connector = "_"
        first_section = connector.join(parts[:-1])
       
        filename = first_section + "-" + parts[-1]
    if len(filename.split(".")) > 1:
        parts = filename.split(".")
        connector = ""
        first_section = connector.join(parts[:-1])
        filename = first_section
    if len(filename.split(" ")) > 1:
        connector = ""
        spaceless = filename.split(" ")
        filename = connector.join(spaceless)
    '''
    if len(filename)>33:
        parts = filename.split("-")
        first_section = parts[0][:30]
        filename = first_section+"-"+ parts[1]
    '''
    return filename


def inputs():
    if len(sys.argv) > 1:
        print(sys.argv[1])
        try:
            input_string = str(sys.argv[1]).split("_")
            ratio1 = input_string[0]
            ratio2 = input_string[1]
            ratio = ratio1+"/"+ratio2
            width = eval(input_string[2])
            interval = eval(input_string[3])
            remainder = eval(input_string[4])
            frame_rate = eval(input_string[5])
            focus = input_string[6]
            max_loops = eval(input_string[7])
            if sys.argv[2] == "True":
                replace = True
            else:
                replace = False

        except:
            print("Error in input string: using default settings")
    else:

        ratio = "4/3"
        width = 500
        interval = 5
        remainder = 3
        frame_rate = 10
        focus = ["C"]
        max_loops = 10
        replace = False

    print(replace)
    return ratio, width, interval, remainder, frame_rate, focus, max_loops, replace

    
if __name__ == "__main__":
    start = time.time()

    ratio, width, interval, remainder, frame_rate, focus, max_loops, replace = inputs()

    # Program processes all videos in directory folder
    directory = os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "wind footage 2")
    ratio_split = ratio.split("/")
    save_directory = os.path.join(os.path.join(os.path.split(os.path.abspath(os.curdir))[0], "Frames"), "{}_{}_{}_{}_{}_{}_{}_{}".format(str(ratio).split("/")[0], str(ratio).split("/")[1], str(width),str(interval),str(remainder),str(frame_rate),str("".join(focus)), str(max_loops)))
    ratio = eval(ratio)
    
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    total_videos = 0
    for folder in os.listdir(directory):
        if not "bin" in folder and not "stablise" in folder and not "TODO" in folder: 
            current_path = os.path.join(directory,folder)
            for filename in os.listdir(current_path):
                total_videos +=1

    count = 0

    threads = []
    for folder in os.listdir(directory):
        if not "bin" in folder and not "stablise" in folder and not "TODO" in folder: 
            current_path = os.path.join(directory,folder)
            for filename in os.listdir(current_path):
                
                try:
                    filename_string = cleanid(filename)

                    save_path = os.path.join(save_directory, filename_string)
                    
                     # Setup folder for outputs
                    if os.path.exists(save_path) and replace == True:
                        shutil.rmtree(save_path)
                    elif os.path.exists(save_path) and replace == False:
                        #print("skipping", filename)
                        continue
                    
                    print("Processing", filename)
                    print(save_path)
                    
                    os.mkdir(save_path)
                    
                    for focus_point in focus:
                        while len(threads) >= constants.THREADS:
                            for thread in threads:
                                if not thread.is_alive():   
                                    threads.remove(thread)
                        load_path = os.path.join(current_path, filename)

                        # Due to ratio reduction there needs only be 2 corners focused on. 
                        # Ratio is reduced in such a way that only one dimension is reduced at a time. 
                        
                        p1 = multiprocessing.Process(target=process_video, args=(load_path, save_path, filename_string, focus_point, ratio, width, interval, remainder, frame_rate, max_loops))
                        threads.append(p1)
                        p1.start()
                   
                    count +=1 
                    if count % 10 == 0:
                        print("Completed {} out of {} folders".format(count, total_videos))
                
                except Exception as err:
                    print("ERROR. Processing for file", filename, "failed with error",err)
            
        else:
            print("skipping", folder)

    for thread in threads:
        thread.join()

    end=time.time()
    print("Moment of truth...")
    print("Time taken:")
    print(str(end - start))

import requests, requests.auth
import zipfile
import os
import csv
import time

def download_url(url, save_path, chunk_size=128):
    zip_path = os.path.join(save_path, "all.zip")
    r = requests.get(url, stream=True, auth=requests.auth.HTTPBasicAuth('craney', 'TokenPassword212'))
    print("Downloading")
    with open(zip_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

    print("Download Complete")


def download_weather(url, save_path, chunk_size=128):
    weather_path = os.path.join(save_path, "weather.csv")
    r = requests.get(url, stream=True, auth=requests.auth.HTTPBasicAuth('craney', 'TokenPassword212'))
    with open(weather_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    

def extract(save_path):
    '''
    extracts files if they are not already present
    '''

    current_files = os.listdir(save_path)

    zip_path = os.path.join(save_path, "all.zip")
    archive = zipfile.ZipFile(zip_path, "r")
    download_items = archive.namelist()

    
    for item in download_items:
        name = item.split(".")[0]
        downloaded = False
        for stored_item in current_files:
            
            if name in stored_item.split(".")[0]:
                downloaded = True
                break
        if not downloaded:
            archive.extract(item, save_path)
        else:
            print("item", item, "already downloaded")


def annotate_videos(path):
    weather_path = os.path.join(path, "weather.csv")
    weather_data = read_weather(weather_path)

    for dir_file in os.listdir(path):
        if dir_file.split(".")[-1] == "avi":
            if "--" not in dir_file:# if not labelled... 
                video_time =dir_file.split(".")[0]
                
                video_epoch_time = time.mktime(time.strptime(video_time, '%Y-%m-%d %H-%M-%S'))
                time_differences = {}
                for key in weather_data.keys():
                    key_epoch_time = int(key)
                    difference = abs(key_epoch_time - video_epoch_time) #closest time
                    time_differences[key] = difference

                if list(time_differences.values()) != []:

                    minimum = min(time_differences, key=time_differences.get)
                    print(minimum)
                    if time_differences[minimum] >= 3000: # if over an hour (actually a bit under and hour: hour is 3600)
                        print("no link for video", video_time )
                    else:
                        print("linked video time {} with key time {}".format(video_time, time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(int(minimum))) ))
                        original_file = os.path.join(path,dir_file)
                        name = dir_file.split(".")[0]
                        extension = dir_file.split(".")[1]

                        
                        wind_speed = round(eval(weather_data[minimum][0])) # avg windspeed, ROUNDED to int
                        new_name = name +"--"+str(wind_speed)+"."+extension
                        print(new_name)
                        
                        os.rename(original_file, os.path.join(path, new_name))

                        #time_differences[key] = time_stamp - date_time
                else:
                    print("no appropriate weather data for video", video_time)


                # Testing
                # 1. Does rename just make a new file or replace the old one?
                # 2. Does extraction prevention work
                # 3. Delete zip once done
         



def read_weather(csv_path):
    weather_data = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            row = row[0].split(",")
            weather_data[row[3]] = [row[2], row[1]] #avg, gust
    
    return weather_data



if __name__ == "__main__":
    download_url("http://samsga.me/craney/all.zip",  "V:\\Uni4\\SoloProject\\home_camera")
    download_weather("http://samsga.me/craney/weather.csv",  "V:\\Uni4\\SoloProject\\home_camera")
    print("Extracting")
    extract("V:\\Uni4\\SoloProject\\home_camera")
    annotate_videos("V:\\Uni4\\SoloProject\\home_camera")
    
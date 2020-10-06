import os
import errno, os, stat, shutil

def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
      func(path)
  else:
      raise




def extensionCorrection():
    dir ="F:\\Uni3\\Project\\data"
    for filename in os.listdir(dir):
        os.rename(os.path.join(dir,filename), os.path.join(dir,filename.split(".")[0]))

def denseFlowRemoval():
    dir ="F:\\Uni3\\Project\\data"
    for filename in os.listdir(dir):
        path = dir + "\\" + filename
        for file in os.listdir(path):
            if "DenseFlow" in file:
                print("Removing", path + "\\" + file)
                shutil.rmtree(path + "\\" + file, ignore_errors=False, onerror=handleRemoveReadonly)

def dotRemoval():
    dir ="V:\\uni3\\project\\Frames"
    for filename in os.listdir(dir):
        new_filename =filename
        print(filename)
        if len(filename.split(".")) > 1:
            parts = filename.split(".")
            connector = ","
            first_section = connector.join(parts[:-1])
            new_filename = first_section
            print(new_filename)
        os.rename(os.path.join(dir,filename),os.path.join(dir,new_filename))

def spaceRemoval():
    dir ="V:\\uni3\\project\\Frames"
    for filename in os.listdir(dir):
        print(filename)
        new_filename = filename
        if len(filename.split(" ")) > 1:
            connector = ""
            spaceless = filename.split(" ")
            new_filename = connector.join(spaceless)
            print(new_filename)
        os.rename(os.path.join(dir,filename),os.path.join(dir,new_filename))

def PointsFolderRemover():
    dir ="V:\\uni3\\project\\OpticalFlow\\"
    for folder in os.listdir(dir):
        for flow_folder in os.listdir(dir+folder):
            if flow_folder == "PointsFlow":
                print("removing")
                shutil.rmtree(dir+folder+"\\"+flow_folder)

def folderMaker():
    dir ="V:\\uni3\\project\\OpticalFlow\\"
    for folder in os.listdir(dir):
        if not os.path.exists(os.path.join(dir, folder, "TracksFlow")):
            print("added")
            os.mkdir(os.path.join(dir, folder, "TracksFlow"))

print("starting...")
folderMaker()

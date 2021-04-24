import cv2
import datetime
import time
import schedule
from ftplib import FTP

def capture_video():
    cap = cv2.VideoCapture(0)
    print(cap.get(5))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


    filename = datetime.datetime.strftime(datetime.datetime.now(), "Special_Capture_%Y-%m-%d %H-%M-%S.avi")
    print("making",filename)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename,fourcc, 10.0, (1920,1080))


    
    index = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True and index % 3 == 0:
            out.write(frame)
        index+=1
        if index > 30 * 30: # 30 seconds clips
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    upload_video(filename)

def upload_video(filename):
    print("Uploading", filename)
    session = FTP('samsga.me','matt','N1njach1cken')
    file = open(filename,'rb')                  # file to send
    session.storbinary('STOR '+ filename, file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print("Upload complete")



if __name__ == "__main__":
    print("Beginning capture")
    capture_video()
    print("Completed capture")

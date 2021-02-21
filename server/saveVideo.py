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


    filename = datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d %H-%M-%S.avi")
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
    #upload_video(filename)

def upload_video(filename):
    print("Uploading", filename)
    session = FTP('samsga.me','matt','N1njach1cken')
    file = open(filename,'rb')                  # file to send
    session.storbinary('STOR '+ filename, file)     # send the file
    file.close()                                    # close file and FTP
    session.quit()
    print("Upload complete")



if __name__ == "__main__":
    capture_video()
    schedule.every().day.at("07:00").do(capture_video)
    schedule.every().day.at("08:00").do(capture_video)
    schedule.every().day.at("09:00").do(capture_video)
    schedule.every().day.at("10:00").do(capture_video)
    schedule.every().day.at("11:00").do(capture_video)
    schedule.every().day.at("12:00").do(capture_video)
    schedule.every().day.at("13:00").do(capture_video)
    schedule.every().day.at("14:00").do(capture_video)
    schedule.every().day.at("15:00").do(capture_video)
    schedule.every().day.at("16:00").do(capture_video)
    schedule.every().day.at("17:00").do(capture_video)
    schedule.every().day.at("18:00").do(capture_video)



    while 1:
        schedule.run_pending()
        time.sleep(1)
from itertools import count
from cv2 import data
import numpy as np
import cv2
import glob
import os
MHI_DURATION = 40  # 軌跡
DEFAULT_THRESHOLD = 32  # 閾値
Log = 10
WIDTH = 244
HEIGHT = 224
CLASS = ['Background', 'Bear', 'Boar']
MHI_Save_path = r"I:\TwoStreamTest500\MHi"
RGB_Save_path = r"I:\TwoStreamTest500\RGB"
Data_path = r"I:\Special_Research\TwoStreamCNN\TwoStreamDataset_MHI40\UseMovie"

def main():
    data_directory = glob.glob(Data_path+'\*')
    print("DataDirectory: ", data_directory)
    for x, file in enumerate(data_directory):
        files = glob.glob(file+'\*')
        count_file = 0
        for i, movie in enumerate(files):
            print("NowMovie =", movie, " Class =", CLASS[x])
            cam = cv2.VideoCapture(movie)
            ret, frame = cam.read()
            h, w = frame.shape[:2]
            prev_frame = frame.copy()
            motion_history = np.zeros((h, w), np.float32)
            timestamp = 0
            count_frame = 0
            while(cam.isOpened()):
                ret, frame = cam.read()
                if ret == True:
                    frame_diff = cv2.absdiff(frame, prev_frame)
                    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                    ret, fgmask = cv2.threshold(
                        gray_diff, DEFAULT_THRESHOLD, 255, cv2.THRESH_BINARY)
                    timestamp += 1
                    cv2.motempl.updateMotionHistory(
                        fgmask, motion_history, timestamp, MHI_DURATION)
                    motionhistoryImage = np.uint8(np.clip(
                        (motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
                    output = cv2.medianBlur(motionhistoryImage, ksize=3)
                    if count_frame >= MHI_DURATION:
                        if count_file % Log == 0:
                            print(MHI_Save_path+'/' +CLASS[x] + '/' + str(int(count_file/Log)))
                            os.mkdir(MHI_Save_path+'/'+CLASS[x] + '/' +str(int(count_file/Log)))
                            cv2.imwrite((RGB_Save_path+'/'+CLASS[x] + '/' +str(int(count_file/Log))) + '.jpg',frame)
                        cv2.imwrite(MHI_Save_path+'/'+ CLASS[x] +'/'+str(int(count_file/Log)) + '/' + str(count_frame) + '.jpg', output)
                    count_frame += 1
                    count_file +=1
                    prev_frame = frame.copy()
                else:
                    break
            cam.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

import os 
import glob
import cv2
import shutil
import pprint as pp
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

path = r"E:\Research\DataSet\Wild_Life\3rd_Season\Image\train\RGB\train"
save = r"E:\Research\DataSet\Wild_Life\3rd_Season\Image\train\RGB_5000"
num = 5000
classList = os.listdir(path)

for i,classlist in enumerate(sorted(glob.glob(path + "/*"), key=natural_keys)):
    os.mkdir(save+"/"+classList[i])
    for x,Imagelist in enumerate(sorted(glob.glob(classlist + "/*"), key=natural_keys)):
        if x%num == 0:
            print(x,Imagelist)
            shutil.copy(Imagelist,save+"/"+classList[i])

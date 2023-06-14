from ctypes.wintypes import RGB
import enum
import cv2
import glob
from matplotlib.pyplot import cla
import numpy as np
import os
import re
import tqdm
import pprint as pp

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class_num = 6
class make_stackImage():
    def __init__(self):
        self.RGB_Images = [[] for rgb in range(class_num)] 
        self.MHI_Images = [[] for mhi in range(class_num)] 
        self.ClassList = []

    def RGB_List(self,RGB_image_directory):
        for i,Class_file in enumerate(RGB_image_directory):
            print("RGB-"+str(Class_file),i)
            Image_file= glob.glob(Class_file+'\*')
            Image_file.sort(key=natural_keys)
            for file in (Image_file):
                imgs = glob.glob(file+'\*')
                for x,image in enumerate(imgs):
                    self.RGB_Images[i].append(image)
            
        return self.RGB_Images

    def MHI_List(self,MHI_image_directory):
        for i,Class_file in enumerate(MHI_image_directory):
            print("MHI-"+str(Class_file),i)
            Image_file= glob.glob(Class_file+'\*')
            Image_file.sort(key=natural_keys)
            for file in (Image_file):
                imgs = glob.glob(file+'\*')
                for x,image in enumerate(imgs):
                    self.MHI_Images[i].append(image)
        return self.MHI_Images
    
    def Synthetic_Image(self,RGB_Path_List,MHI_Path_List,Synthetic_Image_Directory,DisplayFlag):
        for x in range(class_num):
            os.makedirs(str(Synthetic_Image_Directory+"//"+self.ClassList[x]))
            print(self.ClassList[x])
            for i,RGB_path in enumerate(RGB_Path_List[x]):
                RGB_image = cv2.imread(RGB_path)
                MHI_image = cv2.imread(MHI_Path_List[x][i])
                Synthetic_Image = cv2.addWeighted(src1=RGB_image,alpha=1,src2=MHI_image,beta=0.5,gamma=0)
                if DisplayFlag == True:
                    cv2.imshow("Synthetic_Image",Synthetic_Image)
                    cv2.waitKey(10)
                cv2.imwrite(Synthetic_Image_Directory + "//"+self.ClassList[x]+"//" "Co-Occurrence_Image_" + str(i) + ".jpg",Synthetic_Image)


    def __call__(self,RGB_Save_Directory=None,MHI_Save_Directory=None,Synthetic_Image_Directory = None,DisplayFlag = False):
        RGB_Images = self.RGB_List(glob.glob(RGB_Save_Directory+'\*'))
        MHI_Images = self.MHI_List(glob.glob(MHI_Save_Directory+'\*'))
        self.ClassList = os.listdir(RGB_Save_Directory)
        self.Synthetic_Image(RGB_Images,MHI_Images,Synthetic_Image_Directory,DisplayFlag)



test = make_stackImage()
test(RGB_Save_Directory=r"",
    MHI_Save_Directory=r"",
    Synthetic_Image_Directory = r"")




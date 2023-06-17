from ctypes.wintypes import RGB
import cv2
import glob
import numpy as np
import re
import pprint as pp
from sklearn.linear_model import enet_path

class make_stackImage():
    def __init__(self,RGB_Images = [],MHI_Images = []):
        self.RGB_Images = RGB_Images
        self.MHI_Images = MHI_Images

    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        return [ self.atoi(c) for c in re.split(r'(\d+)', text) ]

    def RGB_List(self,RGB_image_directory):
        for i,RGB_folders in enumerate((RGB_image_directory)):
            for x,RGB_image in enumerate(glob.glob(RGB_folders + "/*")):
                self.RGB_Images.append(RGB_image)
        return self.RGB_Images

    def MHI_List(self,MHI_image_directory):
        for i,MHI_folders in enumerate(MHI_image_directory):
            for x,MHI_folder in enumerate(glob.glob(MHI_folders + "/*" )):
                MHI_image = (glob.glob(MHI_folder + "/*" ))
                self.MHI_Images.append(MHI_image[-1])
        return self.MHI_Images
    
    def Synthetic_Image(self,RGB_Path_List,MHI_Path_List,Synthetic_Image_Directory,DisplayFlag=False):
        for i,(RGB_path,MHI_path) in enumerate(zip(RGB_Path_List,MHI_Path_List)):
            RGB_image = cv2.imread(RGB_path)
            MHI_image = cv2.imread(MHI_path)
            Synthetic_Image = cv2.addWeighted(src1=RGB_image,alpha=1,src2=MHI_image,beta=0.5,gamma=0)
            #if DisplayFlag == True:
            #    cv2.imshow("Synthetic_Image",Synthetic_Image)
            #    cv2.waitKey(10)
            cv2.imwrite(Synthetic_Image_Directory + "//" + "Synthetic_Image_" + str(i) + ".jpg",Synthetic_Image)

        print("fin")

    def __call__(self,RGB_Save_Directory=None,MHI_Save_Directory=None,Synthetic_Image_Directory = None,DisplayFlag = False):
        RGB_Images = self.RGB_List(glob.glob(RGB_Save_Directory+'\*'))
        MHI_Images = self.MHI_List(glob.glob(MHI_Save_Directory+'\*'))
        self.Synthetic_Image(RGB_Images,MHI_Images,Synthetic_Image_Directory,DisplayFlag)

test = make_stackImage()
test(RGB_Save_Directory=r"",
    MHI_Save_Directory=r"",
    Synthetic_Image_Directory = r"")

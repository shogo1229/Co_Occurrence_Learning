from ast import Pass
import random
import glob
import pprint as pp
import shutil
import os
import re
from tqdm import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class extractImage():
    def __init__(self,RGBpath,CoPath):
        self.classList = (os.listdir(RGBpath))
        self.extracts = 500
        self.RGB_ImageList = [[],[],[]]
        self.Co_ImageList = [[],[],[]]
        self.num_List = []
    def random_num(self,lower_limit,upper_limit,extracts_Number):
        while len(self.num_List) < extracts_Number:
            random_num = random.randint(lower_limit,upper_limit)
            if (not random_num in self.num_List) and (random_num < upper_limit):
                self.num_List.append(random_num)
        return self.num_List
    def extracts_Image(self,rgb,Co,RGB_savePath,Co_savePath,idx):
        print(self.classList[idx])
        os.mkdir(RGB_savePath+'/'+self.classList[idx])
        os.mkdir(Co_savePath+'/'+self.classList[idx])
        rgb.sort(key=natural_keys)
        Co.sort(key=natural_keys)
        extractsList = self.random_num(0,(len(rgb)-1000),self.extracts)
        for Image_id in tqdm(range(self.extracts)):
            #print(str(rgb[extractsList[Image_id]]))
            #print(str(Co[extractsList[Image_id]]))
            #print("-------------------------------------------------")
            shutil.copy(str(rgb[extractsList[Image_id]]),RGB_savePath+'/'+self.classList[idx])
            shutil.copy(str(Co[extractsList[Image_id]]),Co_savePath+'/'+self.classList[idx])
    def __call__(self,RGB_Images,Co_Images,RGB_SavePath,Co_SavePath):
        RGB_classList = glob.glob(RGB_Images + "\*")
        Co_classList = glob.glob(Co_Images + "\*")
        for idx,(rgb,Co) in enumerate(zip(RGB_classList,Co_classList)):
            rgb_folderList = glob.glob(rgb + "\*")
            Co_folderList =  glob.glob(Co + "\*")
            self.extracts_Image(rgb_folderList,Co_folderList,RGB_SavePath,Co_SavePath,idx)
if __name__ == '__main__':
    test = extractImage(RGBpath=r"E:\Research\DataSet\20BN-Jester-6Class\6class_RGB\test",
                    CoPath=r"E:\Research\DataSet\20BN-Jester-6Class\6class_Synthetic\test")

    test(RGB_Images=r"E:\Research\DataSet\20BN-Jester-6Class\6class_RGB\test",
    Co_Images=r"E:\Research\DataSet\20BN-Jester-6Class\6class_Synthetic\test",
    RGB_SavePath=r"E:\Research\DataSet\20BN-Jester-6Class_Grad\GradTest500_RGB",
    Co_SavePath=r"E:\Research\DataSet\20BN-Jester-6Class_Grad\GradTest500_Syntetic")

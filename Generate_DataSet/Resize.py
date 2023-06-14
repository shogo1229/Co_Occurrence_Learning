import enum
import cv2
from cv2 import resize
import numpy
import os
import glob
from tqdm import tqdm

class Resize():
    def __init__(self,path):
        self.classList = os.listdir(path)
        self.PATH = path
    def resize(self,Folder):
        ImgList = glob.glob(Folder+"\*")
        for z,img in enumerate(ImgList):
            #print(img)
            image = cv2.imread(img)
            image = cv2.resize(image,(224,224))
            cv2.imwrite(str(img),image)

    def __call__(self):
        Folder = glob.glob(self.PATH+"\*")
        for i,f in enumerate(Folder):
            #print(self.classList[i])
            #print(f)
            #Flist = glob.glob(f+"\*")
            self.resize(f)
            #for x,Folder in enumerate(Flist):
            #    self.resize(Folder)



test = Resize(path=r"")
test()




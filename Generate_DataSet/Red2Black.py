import cv2
import numpy
import os
import glob
import numpy as np

class Red2Black():
    def __init__(self):
        self.classList = []
        self.Red =[0, 0, 224]
        self.Black = [0,0,0]
    def resize(self,path):
        for i,class_file in enumerate(glob.glob(path+'\*')):
            for x,file in enumerate(glob.glob(class_file+'\*')):
                print(file)
                for y,image in enumerate(glob.glob(file+'\*')):
                    img = cv2.imread(image)
                    np.where(img == self.Red,self.Black,img)
                    cv2.imshow("test",img)
                    cv2.waitKey(100)
                    #cv2.imwrite(image,img)
    def __call__(self,path):
        self.classList = os.listdir(path)
        self.resize(path)

test = Red2Black()
test(path =r"")
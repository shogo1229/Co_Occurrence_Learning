import enum
import imp
import pprint as pp
import os 
import glob
import shutil
from tkinter.tix import Tree
from matplotlib import image
from sklearn.linear_model import enet_path
from sympy import true
from tqdm import tqdm

class Moves():
    def __init__(self,path):
        self.Image_Class = self.Image_Class = os.listdir(path)
    def move_image(self,ClassList):
        for i,Class in enumerate(ClassList):
            List = glob.glob(Class + "\*")
            for x,imgs in enumerate(List):
                img = glob.glob(imgs+ "\*")
                for y,file in enumerate(img):
                    shutil.move(file,Class)
    def move_folder(self,ClassList,folder_path):
        for i,Class in enumerate(tqdm(ClassList)):
            List = glob.glob(Class + "\*")
            print(Class)
            for x,folder in enumerate(List):
                shutil.move(folder,folder_path)

    def __call__(self,folder_Path=None,move_image_flag=True, move_folder_flag=False):
        ClassList = glob.glob(folder_Path + "\*")
        if move_image_flag == True:
            self.move_image(ClassList)
        elif move_folder_flag == True:
            self.move_folder(ClassList,folder_Path)


path = r"E:\Research\DataSet\Wild_Life\4th_Season\Image\test\RGB"
test = Moves(path)
test(path)

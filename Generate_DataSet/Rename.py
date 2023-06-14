import os 
import glob
from tkinter.tix import Tree
from matplotlib import image
from tqdm import tqdm

class Rename():
    def __init__(self,path):
        self.Image_Class = os.listdir(path)
    def Rename_folder(self,folder_List,Class,i):
        for x,folder in enumerate(folder_List):
            newpath = (str(Class) + '\\v_' + str(self.Image_Class[i]) + '_g' + str(x) + "_c1")
            os.rename(folder,newpath)
    def Rename_Image(self,folder_List,Class,i):
        for x,image in enumerate(folder_List):
            #print(image)
            Image_List = glob.glob(image + "\*")
            for y,img in enumerate(Image_List):
                newpath = (str(image) + '\\frame'+ str(y).zfill(6) + ".jpg")
                os.rename(img,newpath)

    def __call__(self,folder_Path=None,Rename_Folder =False,Rename_Image =True):
        ClassList = glob.glob(folder_Path + "\*")
        for i,Class in enumerate(tqdm(ClassList)):
            folder_List = glob.glob(Class + "\*")
            if Rename_Folder == True:
                self.Rename_folder(folder_List,Class,i)
            elif Rename_Image == True:
                self.Rename_Image(folder_List,Class,i)

path = r""
test = Rename(path)
test(folder_Path= path,Rename_Folder=True,Rename_Image=False)
test(folder_Path= path,Rename_Folder=False,Rename_Image=True)


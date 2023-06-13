import os
import sys
import glob
sys.path.append('../../')
from Motion_history_image.Motion_History_Image_read_image import MotionHistoryImage

class make_TwoStream_Data():
    def __init__(self):
        self.MHI = MotionHistoryImage()
    def make_DataSet(self,Movie_Path = None,MHI_Save_Path = None,RGB_Save_Path = None,RGB_split_Save_Path = None):
        os.makedirs(str(RGB_Save_Path))
        os.makedirs(str(MHI_Save_Path))
        self.MHI(ImageList=Movie_Path,RGB_SavePath=RGB_Save_Path,MHI_SavePath=MHI_Save_Path,RGB_split_SavePath=RGB_split_Save_Path,SaveFlag=True)
    def __call__(self,Movie_path,RGB_Save_Directory=None,MHI_Save_Directory=None,RGB_split_Save_Directory = None):
        data_directory = glob.glob(Movie_path+'\*')
        for i,file in enumerate(data_directory):
            print("Class Name :",(data_directory[i].replace(str(Movie_path),"")).lstrip('\\'))
            files = glob.glob(file+'\*')
            for x,movie in enumerate(files):
                RGB = str(RGB_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'RGB_' + str(x)
                MHI = str(MHI_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'MHI_' + str(x)
                self.make_DataSet(Movie_Path=movie,RGB_Save_Path=RGB,MHI_Save_Path=MHI)

make_data = make_TwoStream_Data()
make_data(Movie_path=r"E:\Research\DataSet\20BN-Jester-6Class\6class_org\test",
        RGB_Save_Directory=r"C:\CaclAcc\RGB",
        MHI_Save_Directory=r"C:\CaclAcc\MHI"
        )
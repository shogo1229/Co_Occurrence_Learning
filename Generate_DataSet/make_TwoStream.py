import imp
import os
import sys
import glob
sys.path.append('../')
from tqdm import tqdm
from Motion_history_image.Motion_History_Image import MotionHistoryImage
from Motion_history_image.Motion_History_Image_prev5 import MotionHistoryImage_prev5
from Motion_history_image.Motion_History_Image_Pseudo_color import MotionHistoryImage_Pseudo

class make_TwoStream_Data():
    def __init__(self):
        self.MHI = MotionHistoryImage()
    def make_DataSet(self,Movie_Path = None,MHI_Save_Path = None,RGB_Save_Path = None,CoOccurrence_Save_Path = None):
        os.makedirs(str(RGB_Save_Path))
        os.makedirs(str(MHI_Save_Path))
        os.makedirs(str(CoOccurrence_Save_Path))
        self.MHI(MoviePath=Movie_Path,RGB_SavePath=RGB_Save_Path,MHI_SavePath=MHI_Save_Path,CoOccurrence_SavePath=CoOccurrence_Save_Path,SaveFlag=True)
    def __call__(self,Movie_path,RGB_Save_Directory=None,MHI_Save_Directory=None,CoOccurrence_Save_Directory = None):
        data_directory = glob.glob(Movie_path+'\*')
        for i,file in enumerate(data_directory):
            print("Class Name :",(data_directory[i].replace(str(Movie_path),"")).lstrip('\\'))
            files = glob.glob(file+'\*')
            for x,movie in enumerate(tqdm(files)):
                RGB = str(RGB_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'RGB_' + str(x)
                MHI = str(MHI_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'MHI_' + str(x)
                CoOccurrence = str(CoOccurrence_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'CoOccurrence_' + str(x)
                self.make_DataSet(Movie_Path=movie,RGB_Save_Path=RGB,MHI_Save_Path=MHI,CoOccurrence_Save_Path=CoOccurrence)

make_data = make_TwoStream_Data()
make_data(Movie_path=r"F:\AROB_Journal\20BN-Jester_6Class\Movie\test",
        RGB_Save_Directory=r"F:\AROB_Journal\Normal_MHI_log5\test\Spatial",
        MHI_Save_Directory=r"F:\AROB_Journal\Normal_MHI_log5\test\Temporal",
        CoOccurrence_Save_Directory=r"F:\AROB_Journal\Normal_MHI_log5\test\Co-Occurrence"
        )
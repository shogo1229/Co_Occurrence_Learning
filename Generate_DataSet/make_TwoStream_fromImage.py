import os
import sys
import glob
import imp
import os
import sys
import glob
sys.path.append('../')
from tqdm import tqdm
from Motion_history_image.Motion_History_Image_read_image import MotionHistoryImage

class make_TwoStream_Data():
    def __init__(self):
        self.MHI = MotionHistoryImage()
    def make_DataSet(self,List = None,MHI_Save_Path = None,RGB_Save_Path = None,CoOccurrence_Save_Path = None):
        #os.makedirs(str(RGB_Save_Path))
        os.makedirs(str(MHI_Save_Path))
        os.makedirs(str(CoOccurrence_Save_Path))
        self.MHI(ImageList=List,RGB_SavePath=RGB_Save_Path,MHI_SavePath=MHI_Save_Path,CoOcc_SavePath=CoOccurrence_Save_Path,SaveFlag=True)
    def __call__(self,Movie_path,RGB_Save_Directory=None,MHI_Save_Directory=None,CoOcc_Save_Directory=None):
        data_directory = glob.glob(Movie_path+'\*')
        for i,file in enumerate(data_directory):
            print("Class:"+str(file))
            files = glob.glob(file+'\*')
            for x,ImageFiles in enumerate(tqdm(files)):
                ImageList = glob.glob(ImageFiles+'\*')
                RGB = str(RGB_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'RGB_' + str(x)
                MHI = str(MHI_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'MHI_' + str(x)
                CoOcc = str(CoOcc_Save_Directory) + '/' + ((data_directory[i].replace(str(Movie_path),"")).lstrip('\\')) + '/' + 'CoOcc_' + str(x)
                self.make_DataSet(List=ImageList,RGB_Save_Path=RGB,MHI_Save_Path=MHI,CoOccurrence_Save_Path=CoOcc)

make_data = make_TwoStream_Data()
make_data(Movie_path=r"I:\AROB_Journal\20BN-Jester_6Class\test",
        RGB_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\test\RGB",
        MHI_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\test\MHI",
        CoOcc_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\test\CoOcc"
        )

make_data = make_TwoStream_Data()
make_data(Movie_path=r"I:\AROB_Journal\20BN-Jester_6Class\val",
        RGB_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\val\RGB",
        MHI_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\val\MHI",
        CoOcc_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\val\CoOcc"
        )

make_data = make_TwoStream_Data()
make_data(Movie_path=r"I:\AROB_Journal\20BN-Jester_6Class\train",
        RGB_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\train\RGB",
        MHI_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\train\MHI",
        CoOcc_Save_Directory=r"I:\AROB_Journal\Normal_MHI_log5\train\CoOcc"
        )

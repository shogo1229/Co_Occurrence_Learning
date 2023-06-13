import cv2
import glob
import os
from tqdm import tqdm
import moviepy.editor as mp
from sklearn.linear_model import enet_path

class inversion_Movie():
    def __init__(self,Movie_Directory):
        self.movie_Class = os.listdir(Movie_Directory)

    def Movie_path(self,Movie_Directory,Inversion_Save_Directory = None):
        for x,Class_file in enumerate(Movie_Directory):
            print("Class Name :",self.movie_Class[x])
            Inversion_Save_Path = str(Inversion_Save_Directory) +"\\"+ str(self.movie_Class[x])
            os.mkdir(Inversion_Save_Path)
            Movie_List= glob.glob(Class_file+'\*')
            for i,Movie_Path in enumerate(tqdm(Movie_List)):
                Save_path = str(Inversion_Save_Path +"//" + str(self.movie_Class[x])+"_" +str(i)+'_inversion.mp4')
                self.inversion_Movie(Movie_Path,Inversion_Save_Path = Save_path)

    def inversion_Movie(self,Movie_Path,Inversion_Save_Path):
        video = cv2.VideoCapture(Movie_Path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_of_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        inversion_video = cv2.VideoWriter(Inversion_Save_Path, fourcc, fps, (width,height))
        for i in range(num_of_frame):
            ret, frame = video.read()
            inversion_frame = cv2.flip(frame, 1) 
            inversion_video.write(inversion_frame)
        inversion_video.release()
        video.release()

    def __call__(self,Movie_Directory=None,Inversion_Save_Directory=None):
        self.Movie_path(glob.glob(Movie_Directory+'\*'),Inversion_Save_Directory)

test = inversion_Movie(Movie_Directory=r"E:\trim")
test(Movie_Directory=r"E:\trim",
    Inversion_Save_Directory = r"E:\inversion")
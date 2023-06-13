from pprint import pprint
import sys
import glob
sys.path.append('../')
from Motion_history_image.Motion_History_Image_Pseudo_color import MotionHistoryImage_Pseudo
import cv2
import glob

MHI = MotionHistoryImage_Pseudo()
MHI(DisplayFlag=True,Pseudo_color=True)



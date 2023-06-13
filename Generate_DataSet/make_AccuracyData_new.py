import os 
import glob
import cv2
import shutil
import pprint as pp
import re

from sympy import limit

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

RGB_path = r"E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\val\Synthetic\RGB_full"
MHI_path = r"E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\val\Synthetic\MHI_org"
RGB_save_path =r"E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\RGB"
MHI_save_path =r"E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\MHI"

num = 15

ClassLists = os.listdir(RGB_path)

for i,(RGB_class,MHI_class) in enumerate(zip(sorted(glob.glob(RGB_path + "/*"), key=natural_keys),sorted(glob.glob(MHI_path + "/*"), key=natural_keys))):
    print("---- Class:"+str(ClassLists[i])+" ----")
    os.mkdir(RGB_save_path+"\\"+ClassLists[i])
    os.mkdir(MHI_save_path+"\\"+ClassLists[i])
    count = 0
    for y,(RGB_Image_folder,MHI_Image_folder) in enumerate(zip(sorted(glob.glob(RGB_class + "/*"), key=natural_keys),sorted(glob.glob(MHI_class + "/*"), key=natural_keys)),start=0):
        rgb_count = 1
        for x,(RGB_Image,MHI_Image) in enumerate(zip(sorted(glob.glob(RGB_Image_folder + "/*"), key=natural_keys),sorted(glob.glob(MHI_Image_folder + "/*"), key=natural_keys)),start=0):
            if x == 0:
                #print("Image:",((len(glob.glob(RGB_Image_folder + "/*"))%num)))
                last = 1
            if x % (num) == 0:
                #print("-----------------------------------------------------------------------------")
                #print(x,MHI_Image)
                #print(count)
                print((MHI_save_path+"\\"+ClassLists[i]+"\\"+str(int(count/num))))
                os.mkdir((MHI_save_path+"\\"+ClassLists[i]+"\\"+str(int(count/num))))
            #
            shutil.copy(MHI_Image,(MHI_save_path+"/"+ClassLists[i]+"/"+str(int(count/num))))
            
            if rgb_count % 15 ==0:
                shutil.copy(RGB_Image,RGB_save_path+"/"+ClassLists[i]+"/"+str(int(count/num)) +".jpg")
                if last == (len(glob.glob(RGB_Image_folder + "/*")) - (len(glob.glob(RGB_Image_folder + "/*"))%num))/num:
                    #print("break")
                    count +=1
                    break
                
                last +=1
            count +=1
            rgb_count +=1
            

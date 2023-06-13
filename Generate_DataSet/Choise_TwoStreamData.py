import random
import glob
import pprint
import shutil
import os
import re

rgb_folder_path = r'E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\ALL\RGB'
mhi_folder_path = r'E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\ALL\MHI'
rgb_save_path = r'E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\choise300\RGB'
mhi_save_path = r'E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\choise300\MHI'
classList = []
extracts = 300

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def rand_ints_nodup(lower_limit,upper_limit,extracts_Number):
    num_List = []
    while len(num_List) < extracts_Number:
        random_num = random.randint(lower_limit,upper_limit - 2)
        if not random_num in num_List:
            num_List.append(random_num)
    return num_List


if __name__ == '__main__':  
    classList = os.listdir(rgb_folder_path)
    for i,(RGB_folder,MHI_folder) in enumerate(zip(sorted(glob.glob(rgb_folder_path + "\*"), key=natural_keys),sorted(glob.glob(mhi_folder_path + "\*"), key=natural_keys)),start=0):
        RGB_Image = sorted(glob.glob(RGB_folder + "\*"), key=natural_keys)
        MHI_Images = sorted(glob.glob(MHI_folder + "\*"), key=natural_keys)
        Num = (rand_ints_nodup(1,len(RGB_Image),extracts))
        os.mkdir(rgb_save_path + "\\" + classList[i])
        print(classList[i])
        for x in range(extracts):
            shutil.copy(RGB_Image[Num[x-1]],rgb_save_path + "\\" + classList[i])
            shutil.copytree(MHI_Images[Num[x-1]],mhi_save_path + "\\" + classList[i] + "\\" + str(Num[x-1]))
            #print(RGB_Image[Num[x-1]],rgb_save_path + "\\" + classList[i])
            #print(MHI_Images[Num[x-1]],mhi_save_path + "\\" + classList[i])
            None
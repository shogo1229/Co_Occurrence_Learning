import random
import glob
import pprint as pp
import shutil
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class extractImage():
    def __init__(self):
        self.classList = []
        self.num_List = []
        self.extracts = 5000
        self.extracts_img =[0]+[1]+[2]

    def random_num(self,lower_limit,upper_limit,extracts_Number):
        while len(self.num_List) < extracts_Number:
            random_num = random.randint(lower_limit,upper_limit)
            if not random_num in self.num_List:
                self.num_List.append(random_num)
        return self.num_List

    def extracts_Image(self,image_path,save_path,i):
        print(self.classList[i])
        files = glob.glob(image_path+'\*')
        os.makedirs(save_path + "\\" + str(self.classList[i]))
        self.extracts_img[i] = self.random_num(0,len(files),self.extracts)
        #self.extracts_img[i] = random.sample(range(0,len(files)), k=self.extracts)
        pp.pprint(self.extracts_img[i])
        for x in range(self.extracts):
            print((files[int(self.extracts_img[i][x])],save_path+'/'+self.classList[i]))
            b = (files[int(self.extracts_img[i][x])])
            shutil.copy(str(b),save_path+'/'+self.classList[i])

    def __call__(self,folder_path = None,Save_directory = None):
        self.classList = os.listdir(folder_path)
        pp.pprint(self.classList)
        dataset_list = glob.glob(folder_path + "\*")
        for i,image_class in enumerate(dataset_list):
            self.extracts_Image(image_class,Save_directory,i)

test = extractImage()
test(folder_path=r"E:\Research\DataSet\Wild_Life\3rd_Season\Image\train\RGB\train"
    ,Save_directory=r"E:\Research\DataSet\Wild_Life\3rd_Season\Image\train\RGB_5000")


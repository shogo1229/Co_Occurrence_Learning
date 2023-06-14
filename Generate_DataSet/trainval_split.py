import glob
import pprint
import shutil
import os
from sklearn.model_selection import train_test_split

class trainval_split():
    def __init__(self,folder_path):
        self.classList = os.listdir(folder_path)

    def classlist(self,data_list,folder_path):
        for i,class_name in enumerate(data_list):
            print(class_name)
            print(folder_path)
            class_name = str(class_name).replace(folder_path + "\\",'')
            self.classList.append(class_name)

    def train_val_split(self,dataset_path,split_ratio):
        files = glob.glob(dataset_path+'\*')
        train_list,val_list = train_test_split(files,train_size=split_ratio, random_state=1)
        return(train_list,val_list)

    def file_copy(self,image_list,save_path,i):
        os.makedirs(save_path + "\\" + str(self.classList[i]))
        for image_path in image_list:
            shutil.move(image_path,(save_path + "\\" + str(self.classList[i])))

    def __call__(self,train_save_directory=None,val_save_directory=None,folder_path=None,split_ratio=0.7):
        dataset_list = glob.glob(folder_path + "\*")
        self.classlist(dataset_list,folder_path)
        for i,image_class in enumerate(dataset_list):
            train_list,val_list = self.train_val_split(image_class,split_ratio)
            self.file_copy(train_list,train_save_directory,i)
            self.file_copy(val_list,val_save_directory,i)
        print("done")

split = trainval_split(folder_path=r"")
split(train_save_directory=r"",
        val_save_directory=r"",
        folder_path=r"",
        split_ratio=0.8)







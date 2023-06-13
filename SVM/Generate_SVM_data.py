import os
import sys
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
sys.path.append('../')
from natsort import natsorted
from Network.Spatial.MobileNet import MobileNet_V2_Spatial
from Network.Temporal.MobileNet import MobileNet_V2_Temporal
from PIL import Image


class BaseTransform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
    def __call__(self, img):
        return self.base_transform(img)

class SVM_Generater():
    def __init__(self,Spatial_data,Temporal_data,Tau=5):
        self.Spatial_data = Spatial_data
        self.Temporal_data = Temporal_data
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.classList = os.listdir(self.Spatial_data)
        self.Tau = Tau
        self.transform = BaseTransform()
        self.count = 0
        self.first_counter = 0
        self.InferencceMHI_Tensor = 0
        self.MHIlist = []
        self.Reset = []
        self.Reset = (torch.tensor(self.Reset)).to(self.device)
        self.check = 0
        self.RGBList = []
        self.SVMList = []
        self.SVM_acc = []

    def calc_Accuracy(self,Spatial_model_path,Temporal_model_path):
        self.Spatial_data = glob.glob(self.Spatial_data+"\*")
        self.Temporal_data = glob.glob(self.Temporal_data+"\*")

        MobileNet_Spatial = MobileNet_V2_Spatial()
        Spatial_model = MobileNet_Spatial.model.cuda()
        Spatial_model.load_state_dict(torch.load(Spatial_model_path))
        Spatial_model.eval()
        MobileNet_Temporal = MobileNet_V2_Temporal()
        Temporal_model = MobileNet_Temporal.model.cuda()
        Temporal_model.load_state_dict(torch.load(Temporal_model_path))
        Temporal_model.eval()

        for Class_idx,Class_folder in enumerate(zip(self.Spatial_data,self.Temporal_data)):
            Spatial_folderList = glob.glob(Class_folder[0]+"\*")
            Temporal_folderList = glob.glob(Class_folder[1]+"\*")
            for folder_idx,folder in enumerate(zip(Spatial_folderList,Temporal_folderList)):
                Spatial_ImageList = glob.glob(folder[0]+"\*")
                Temporal_ImageList = glob.glob(folder[1]+"\*")
                Spatial_ImageList = natsorted(Spatial_ImageList)
                Temporal_ImageList = natsorted(Temporal_ImageList)
                self.first_counter = 0
                self.count = 0
                self.check = 0
                self.MHIlist = []
                for image_idx,image in enumerate(zip(Spatial_ImageList,Temporal_ImageList)):
                    Spatial_Image = Image.open(image[0])
                    Temporal_Image = Image.open(image[1])
                    Temp = image[1]
                    Temporal_Image = Temporal_Image.convert('L')
                    Spatial_tensor = self.transform(Spatial_Image)
                    Temporal_tensor = self.transform(Temporal_Image)
                    Temporal_tensor = Temporal_tensor.unsqueeze_(0)
                    if self.first_counter < self.Tau:
                        self.MHIlist.append(Temporal_tensor)
                        self.RGBList.append(Temp)
                        self.count += 1
                        self.first_counter += 1
                    else:
                        for num in range(self.count - self.Tau, self.count):
                            if num == self.count - self.Tau:
                                InferencceMHI_Tensor = self.MHIlist[num].to(self.device)
                                showMHI_Image = []
                                showMHI_Image.append(self.RGBList[num])
                            else:
                                InferencceMHI_Tensor = torch.cat(((self.MHIlist[num].to(self.device)), InferencceMHI_Tensor), 1)
                                showMHI_Image.append(self.RGBList[num])
                        #print("------------------------------------------------------------------")

                        output_spatial = Spatial_model(Spatial_tensor.unsqueeze(dim=0).to(self.device))
                        output_Temporal = Temporal_model(InferencceMHI_Tensor)
                        Spatial_result = output_spatial.data[0]
                        Temporal_result = output_Temporal.data[0]
                        self.SVMList.append((Spatial_result.tolist()) + (Temporal_result.tolist()))
                        self.SVM_acc.append(Class_idx)
                        self.count += 1
                        self.MHIlist.append(Temporal_tensor)
                        self.RGBList.append(Temp)

    def __call__(self,Spatial_model,Temporal_model,savePath):
        self.calc_Accuracy(Spatial_model,Temporal_model)
        np.savetxt("models/"+str(savePath)+"/" + "explanatory_train.txt", self.SVMList,fmt="%.18f")
        np.savetxt("models/"+str(savePath)+"/" +"objective_train.txt", self.SVM_acc, fmt='%d')

svm_generate = SVM_Generater(Spatial_data=r"E:\Research\DataSet\Wild_Life\Image\RGB\train",
                            Temporal_data=r"E:\Research\DataSet\Wild_Life\Image\MHI\train")

#os.mkdir("models/MobileNet_RGB_Wild-Life_part3_MobileNet_MHI_WildLife_log5_part1")
svm_generate(Spatial_model=r"E:\Research\TwoStreamCNN_2nd-Season\Models\max_Val_acc_MobileNet_RGB_Wild-Life_part3\max_Val_acc_MobileNet_RGB_Wild-Life_part3.pth",
            Temporal_model=r"E:\Research\TwoStreamCNN_2nd-Season\Models\max_Val_acc_MobileNet_MHI_Wild-Life_log5_part1\max_Val_acc_MobileNet_MHI_WildLife_log5.pth",
            savePath="MobileNet_RGB_Wild-Life_part3_MobileNet_MHI_WildLife_log5_part1")

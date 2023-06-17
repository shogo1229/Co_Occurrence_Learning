import enum
import os
import sys
import glob
from cv2 import CirclesGridFinderParameters
import numpy as np
import cv2
import tqdm
import torchvision
import torch
import pprint as pp
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data
sys.path.append('../')
from Network.Spatial.MobileNet import MobileNet_V2_Spatial,MobileNet_V3_small_Spatial
from Network.Temporal.MobileNet import MobileNet_V2_Temporal
from PIL import Image
from itertools import count
from natsort import natsorted
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from IPython.display import display

class SpatialTransform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5])
        ])

    def __call__(self, img):
        return self.base_transform(img)

class TemporalTransform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor()
        ])
    def __call__(self, img):
            return self.base_transform(img)

class Calc_Accuracy_TwoStream():
    def __init__(self,Spatial_data,Temporal_data,Tau=10):
        self.Spatial_data = Spatial_data
        self.Temporal_data = Temporal_data
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.classList = os.listdir(self.Spatial_data)
        self.Tau = Tau
        self.Spatialtransform = SpatialTransform()
        self.Temporaltransform = TemporalTransform()
        self.count = 0
        self.first_counter = 0
        self.InferencceMHI_Tensor = 0
        self.MHIlist = []
        self.Reset = []
        self.Reset = (torch.tensor(self.Reset)).to(self.device)
        self.check = 0
        self.Twostrem_true = []
        self.Spatial_true = []
        self.Temporal_true = []
        self.Twostrem_pred = []
        self.Spatial_pred = []
        self.Temporal_pred = []
        self.RGBList = []

    def print_cmx(self,true,pred,name):
        labels = sorted(list(set(true)))
        columns_labels = [str(self.classList[l]) for l in labels]
        index_labels = [str(self.classList[l]) for l in labels]
        cmx_data = confusion_matrix(true, pred, labels=labels)
        df_cmx = pd.DataFrame(cmx_data, index=index_labels, columns=columns_labels)
        plt.figure(figsize = (20,8))
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        sns.set(font_scale = 1.8)
        sns.heatmap(df_cmx, annot=True,cmap='YlGnBu_r',fmt='g',cbar=False)
        plt.tight_layout()
        plt.savefig(str(name)) 

    def calc_Accuracy(self,Spatial_model_path,Temporal_model_path):
        self.Spatial_data = glob.glob(self.Spatial_data+"\*")
        self.Temporal_data = glob.glob(self.Temporal_data+"\*")

        MobileNet_Spatial = MobileNet_V2_Spatial()
        #MobileNet_Spatial = MobileNet_V3_small_Spatial()
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
                    Spatial_tensor = self.Spatialtransform(Spatial_Image)
                    Temporal_tensor = self.Temporaltransform(Temporal_Image)
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
                        Spatial_result = output_spatial.data[0]
                        output_spatial = F.softmax(output_spatial, dim=1)

                        output_Temporal = Temporal_model(InferencceMHI_Tensor)
                        Temporal_result = output_Temporal.data[0]
                        output_Temporal = F.softmax(output_Temporal,dim=1)

                        pred_idx_Temporal = output_Temporal.max(1)[1]

                        TwoStream_result = Temporal_result + Spatial_result
                        TwoStream_result = TwoStream_result/2
                        output_twostream = F.softmax(TwoStream_result, dim=0)

                        pred_idx_spatial = output_spatial.max(1)[1]
                        pred_idx_Temporal = output_Temporal.max(1)[1]
                        pred_idx_twostream= output_twostream.max(0)[1]

                        self.Spatial_pred.append(pred_idx_spatial[0].data.item())
                        self.Temporal_pred.append(pred_idx_Temporal[0].data.item())
                        self.Twostrem_pred.append(pred_idx_twostream.data.item())

                        #print(pred_idx_Temporal[0].data.item(),Class_idx)
                        #pp.pprint(showMHI_Image)

                        self.count += 1
                        self.MHIlist.append(Temporal_tensor)
                        self.RGBList.append(Temp)

                        self.Twostrem_true.append(Class_idx)
                        self.Spatial_true.append(Class_idx)
                        self.Temporal_true.append(Class_idx)

        print("Spatial_confusion_matrix")
        print(confusion_matrix(self.Spatial_true, self.Spatial_pred))
        print(classification_report(self.Spatial_true, self.Spatial_pred,digits=4))
        self.print_cmx(self.Spatial_true,self.Spatial_pred,"Spatial")
    
        print("Temporal_confusion_matrix")
        print(confusion_matrix(self.Temporal_true, self.Temporal_pred))
        print(classification_report(self.Temporal_true, self.Temporal_pred,digits=4))
        self.print_cmx(self.Temporal_true,self.Temporal_pred,"Temporal")

        print("TwoStream_confusion_matrix")
        print(confusion_matrix(self.Twostrem_true, self.Twostrem_pred))
        print(classification_report(self.Twostrem_true, self.Twostrem_pred,digits=4))
        self.print_cmx(self.Twostrem_true,self.Twostrem_pred,"Twostrem")

    def __call__(self,Spatial_model,Temporal_model):
        self.calc_Accuracy(Spatial_model,Temporal_model)

test = Calc_Accuracy_TwoStream(Spatial_data=r"I:\Wild-Life4th\Normal_MHI_log10\test\RGB",
                                Temporal_data=r"I:\Wild-Life4th\Normal_MHI_log10\test\Color-MHI")

test(Spatial_model=r"I:\TwoStreamCNN_2nd-Season\Models\MobileNet_RGB_Wild-Life4th\MobileNet_RGB_Wild-Life4th.pth",
    Temporal_model=r"I:\TwoStreamCNN_2nd-Season\Models\MobileNet_Color_MHI-tau10_Wild-Life4th\MobileNet_Color_MHI-tau10_Wild-Life4th.pth")
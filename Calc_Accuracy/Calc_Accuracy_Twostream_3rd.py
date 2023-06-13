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
import pickle
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
from natsort import natsorted
import seaborn as sns
import matplotlib.pyplot as plt
import torch.utils.data as data
sys.path.append('../')
from Network.Spatial.MobileNet import MobileNet_V2_Spatial
from Network.Temporal.MobileNet import MobileNet_V2_Temporal
from PIL import Image
from itertools import count
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class BaseTransform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.base_transform(img)

class TestBaseTransform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224, 224]),
        ])

    def __call__(self, img):
        return self.base_transform(img)

class Calc_Accuracy_TwoStream():
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
        self.list = []
        self.Reset = (torch.tensor(self.Reset)).to(self.device)
        self.check = 0
        self.Twostrem_true = []
        self.Spatial_true = []
        self.Temporal_true = []
        self.SVM_true = []
        self.Twostrem_pred = []
        self.Spatial_pred = []
        self.Temporal_pred = []
        self.SVM_pred = []
        self.RGBList = []
        self.SVM_check = []
        self.tt = TestBaseTransform()

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

    def calc_Accuracy(self,Spatial_model_path,Temporal_model_path,SVM_model):
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
        SVM_model = pickle.load(open(SVM_model, 'rb'))

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
                        Spatial_result = output_spatial.data[0]
                        SVM_Spatial = Spatial_result
                        output_spatial = F.softmax(output_spatial, dim=1)

                        output_Temporal = Temporal_model(InferencceMHI_Tensor)
                        Temporal_result = output_Temporal.data[0]
                        SVM_Temporal = Temporal_result
                        output_Temporal = F.softmax(output_Temporal,dim=1)

                        TwoStream_result = Temporal_result + Spatial_result
                        TwoStream_result = TwoStream_result/2
                        output_twostream = F.softmax(TwoStream_result, dim=0)

                        pred_idx_spatial = output_spatial.max(1)[1]
                        pred_idx_Temporal = output_Temporal.max(1)[1]
                        pred_idx_twostream= output_twostream.max(0)[1]

                        SVM_input = ((SVM_Spatial.tolist()) + (SVM_Spatial.tolist()))
                        #print(SVM_input)
                        #SVM_input = [round(SVM_input[n],9)for n in range(len(SVM_input))]
                        
                        self.SVM_check.append(np.array(SVM_input))
                        SVM_input = (np.array(SVM_input))
                        SVM_input = SVM_input[np.newaxis,:]
                        SVM_Result = SVM_model.predict(SVM_input.tolist())
                        #print(SVM_input)
                        #print(SVM_Result)
                        self.Spatial_pred.append(pred_idx_spatial[0].data.item())
                        self.Temporal_pred.append(pred_idx_Temporal[0].data.item())
                        self.Twostrem_pred.append(pred_idx_twostream.data.item())
                        self.SVM_pred.append(SVM_Result[0])


                        self.count += 1
                        self.MHIlist.append(Temporal_tensor)
                        self.RGBList.append(Temp)

                        self.Twostrem_true.append(Class_idx)
                        self.Spatial_true.append(Class_idx)
                        self.Temporal_true.append(Class_idx)
                        self.SVM_true.append(Class_idx)

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

        print("SVM_matrix")
        print(confusion_matrix(self.SVM_true, self.SVM_pred))
        print(classification_report(self.SVM_true, self.SVM_pred,digits=4))
        self.print_cmx(self.SVM_true,self.SVM_pred,"SVM")
        #print(self.SVM_pred)
        np.savetxt("SVM_check.txt",self.SVM_check,fmt="%.18f")
        np.savetxt("SVM_Re.txt",self.SVM_pred,fmt="%d")

    def __call__(self,Spatial_model,Temporal_model,SVM_model):
        self.calc_Accuracy(Spatial_model,Temporal_model,SVM_model)

test = Calc_Accuracy_TwoStream(Spatial_data=r"E:\Research\DataSet\Wild_Life\Image\RGB\test",
                                Temporal_data=r"E:\Research\DataSet\Wild_Life\Image\MHI\test")

test(Spatial_model=r"E:\Research\TwoStreamCNN_2nd-Season\Models\max_Val_acc_MobileNet_RGB_Wild-Life_part1\max_Val_acc_MobileNet_Wild-Life_part1.pth",
    Temporal_model=r"E:\Research\TwoStreamCNN_2nd-Season\Models\max_Val_acc_MobileNet_MHI_Wild-Life_log5_part1\max_Val_acc_MobileNet_MHI_WildLife_log5.pth",
    SVM_model = r"E:\Research\TwoStreamCNN_2nd-Season\SVM\models\MobileNet_Wild-Life_part1_MobileNet_MHI_WildLife_log5_part1\SVM_MobileNet_Wild-Life_part1_MobileNet_MHI_WildLife_log5_part1.sav")
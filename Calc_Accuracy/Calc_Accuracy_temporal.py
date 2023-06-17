from cProfile import label
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import glob
import os
import sys
import cv2
import numpy as np
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append('../')
from itertools import count
from pprint import pprint
from sklearn.metrics import classification_report
from Dataloder.MotionHistory_dataloder import MotionDataset
from Dataloder.MotionHistory_dataloder import MotionHistory_Dataset
#from Dataloder.Pseudo_MotionHistory_dataloder import MotionDataset
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import *
from Network.Temporal.ResNet import resnet50_temporal
from sklearn.metrics import confusion_matrix
from string import digits
from sklearn.utils.multiclass import unique_labels
from Network.Temporal.VGG16 import VGG16_Temporal
from Network.Temporal.MobileNet import MobileNet_V2_Temporal


transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

class Calc_Accuracy():
    def __init__(self,Temporal_folder_path,Temporal_model_path,transforms):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_path = Temporal_model_path
        self.folder_path = Temporal_folder_path
        self.transform = transforms
        self.true = []
        self.pred = []

    def print_cmx(self,true,pred,name):
        labels = sorted(list(set(true)))
        columns_labels = [str([l]) for l in labels]
        index_labels = [str([l]) for l in labels]
        cmx_data = confusion_matrix(true, pred, labels=labels)
        df_cmx = pd.DataFrame(cmx_data, index=index_labels, columns=columns_labels)
        plt.figure(figsize = (15,10))
        sns.heatmap(df_cmx, annot=True,cmap='YlGnBu_r',fmt='g',cbar=False)
        plt.savefig(str(name)) 


    def __call__(self):
        DataLoader = MotionHistory_Dataset(self.folder_path, self.transform,5)
        #MobileNet = MobileNet_V2_Temporal()
        #Temporal_model = MobileNet.model.cuda()
        VGG = VGG16_Temporal()
        Temporal_model = VGG.model.cuda()
        Temporal_model.load_state_dict(torch.load(self.model_path))
        Temporal_model.eval()

        for images,labels in DataLoader:
            self.true.append(labels)
            output_Temporal = F.softmax(Temporal_model((images.unsqueeze(dim=0)).to(self.device))) 
            pred_idx_Temporal = output_Temporal.max(1)[1]
            self.pred.append(pred_idx_Temporal[0].data.item())
        labels = sorted(list(set(self.true) | set(self.pred)))
        print(labels)
        print("Temporal_confusion_matrix")
        print(confusion_matrix(self.true, self.pred,labels))
        print(classification_report(self.true, self.pred,digits=4))
        self.print_cmx(self.true,self.pred,"Temporal")

test = Calc_Accuracy(Temporal_folder_path = r"E:\Research\DataSet\RGB_total_0322\Images\MHI\RGB_total_0322_MHI_val",
                    Temporal_model_path = r"E:\Research\TwoStreamCNN_2nd-Season\wildLife.pth",
                    transforms = transform)
test()


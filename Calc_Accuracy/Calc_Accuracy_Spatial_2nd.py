import imp
from string import digits
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
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import *
from zmq import device
from Network.Spatial.ResNet import resnet50_spatial
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from Network.Spatial.VGG16 import VGG16_Spatial
from Network.Spatial.MobileNet import MobileNet_V2_Spatial
from Network.Spatial.GhostNet import Yuda_GhostNet

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5])
])

class Calc_Accuracy():
    def __init__(self,Spatial_folder_path,Spatial_model_path,transforms):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.Class = os.listdir(Spatial_folder_path)
        self.model_path = Spatial_model_path
        self.folder_path = Spatial_folder_path
        self.transform = transforms
        self.saveFolder = os.path.splitext(os.path.basename(self.model_path))[0]
        self.true = []
        self.pred = []

    def print_cmx(self,true,pred,name):
        labels = sorted(list(set(true)))
        columns_labels = [str(self.Class[l]) for l in labels]
        index_labels = [str(self.Class[l]) for l in labels]
        cmx_data = confusion_matrix(true, pred, labels=labels)
        df_cmx = pd.DataFrame(cmx_data, index=index_labels, columns=columns_labels)
        plt.figure(figsize = (20,8))
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        sns.set(font_scale = 1.8)
        sns.heatmap(df_cmx, annot=True,cmap='YlGnBu_r',fmt='g',cbar=False)
        plt.tight_layout()
        plt.savefig(str(name)+"/"+str(name))

    def __call__(self):
        print(self.Class)
        trainLoader = datasets.ImageFolder(root=self.folder_path, transform=self.transform)
        MobileNet = MobileNet_V2_Spatial()
        Spatial_model = MobileNet.model.cuda()
        Spatial_model.load_state_dict(torch.load(self.model_path))
        Spatial_model.eval()
        os.mkdir(self.saveFolder)

        for images,labels in trainLoader:
            self.true.append(labels)
            output_spatial = F.softmax(Spatial_model((images.unsqueeze(dim=0)).to(self.device)))
            pred_idx_spatial = output_spatial.max(1)[1]
            self.pred.append(pred_idx_spatial[0].data.item())
        labels = sorted(list(set(self.true) | set(self.pred)))
        print(labels)
        print("Spatial_confusion_matrix")
        print(confusion_matrix(self.true, self.pred,labels))
        print(classification_report(self.true, self.pred,digits=4))
        self.print_cmx(self.true,self.pred,self.saveFolder)

test = Calc_Accuracy(Spatial_folder_path = r"E:\Research\DataSet\20BN-Jester-6Class\6class_Synthetic\test",
                    Spatial_model_path = r"E:\Research\TwoStreamCNN_2nd-Season\Models\max_Val_acc_MobileNet_Co-Occurrence_20BN-Jester_6class_part1\max_Val_acc_MobileNet_Co-Occurrence_20BN-Jester_6class_part1.pth",
                    transforms = transform)
test()


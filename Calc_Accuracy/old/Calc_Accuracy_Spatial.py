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
from itertools import count
from pprint import pprint
from sklearn.metrics import classification_report
from torchvision import datasets, transforms, models
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.models import *
sys.path.append('../')
from Network.Spatial.ResNet import resnet50_spatial
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from Network.Spatial.VGG16 import VGG16_Spatial
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


Data_num = 100
Spatial_model_path = r'E:\Research\DataSet\Wild_Life\Image\MHI\test'
Spatial_folder_path = r"E:\Research\DataSet\Wild_Life\Image\RGB\test"


Class = os.listdir(Spatial_model_path)
print(Class)

Spatial_true_multi = [0]*2361 + [1]*4011 + [2]*3379 
def print_cmx(y_true, y_pred,name):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=True,cmap='YlGnBu_r',fmt='g' )
    plt.savefig(str(name)) 
def Spatial_Accuracy():
    Spatial_pred_multi = []
    Spatial_result = []
    Spatial_missmatch_resulet = []
    Spatial_missmatch_num = []
    #Spatial_model = resnet50_spatial(channel=3).to(device)

    VGG=VGG16_Spatial()
    Spatial_model = VGG.model.cuda()
    Spatial_model.load_state_dict(torch.load(Spatial_model_path))
    Spatial_model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5,0.5,0.5],
            std=[0.5,0.5,0.5])
    ])

    Base_files = glob.glob(Spatial_folder_path+"\*")
    for x, file in enumerate(Base_files):
        image_file = glob.glob(file+'\*')
        for y, files in enumerate(image_file):
            testImage = Image.open(files)
            testImageTensor = (transform((testImage))).unsqueeze(dim=0)
            testImageTensor = testImageTensor.to(device)
            output_spatial = F.softmax(Spatial_model(testImageTensor))         #softmax Ver
            #output_spatial = Spatial_model(testImageTensor)
            pred_idx_spatial = output_spatial.max(1)[1]
            #if x != pred_idx_spatial:
                #Spatial_missmatch_resulet.append(Class[pred_idx_spatial])
                #cv2.imwrite((Class[pred_idx_spatial]) + '_' + str(y) + '.jpg',files)
            Spatial_missmatch_resulet.append(Class[pred_idx_spatial])
            Spatial_result.append(output_spatial.data[0])
            Spatial_pred_multi.append(pred_idx_spatial[0].data.item())
    return Spatial_pred_multi, Spatial_result, Spatial_missmatch_resulet

if __name__ == '__main__':
    Spatial_pred_multi, Spatial_result, Spatial_missmatch_resulet = Spatial_Accuracy()

    print("Spatial_confusion_matrix")
    print(confusion_matrix(Spatial_true_multi, Spatial_pred_multi))
    print(classification_report(Spatial_true_multi, Spatial_pred_multi,digits=4))
    print_cmx(Spatial_true_multi, Spatial_pred_multi,"Spatial")


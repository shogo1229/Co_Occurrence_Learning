
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
from Network.Temporal.ResNet import resnet50_temporal
from sklearn.metrics import confusion_matrix
from string import digits
from sklearn.utils.multiclass import unique_labels
from Network.Temporal.VGG16 import VGG16_Temporal

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

Data_num = 50
Temporal_model_path = r""
folder_path = r""
Class = os.listdir(Temporal_model_path)

y_true_multi = [0]*Data_num + [1]*Data_num + [2]*Data_num
y_pred_multi = []
log = 9
reset_tensor = []
mhi_image = []
mhi_image = (torch.tensor(mhi_image)).to(device)
reset_tensor = (torch.tensor(reset_tensor)).to(device)


if __name__ == '__main__':
    #temporal_model = resnet50_temporal(3).to(device)
    VGG = VGG16_Temporal()
    temporal_model = VGG()
    temporal_model.load_state_dict(torch.load(Temporal_model_path))
    temporal_model.eval()

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    Base_files = glob.glob(folder_path+"\*")

    for file in Base_files:
        relay_file = glob.glob(file+"\*")
        for file in relay_file:
            image_file = glob.glob(file+"\*")
            # print(file)
            for i, file in enumerate(image_file):
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                img = Image.fromarray(img)
                img_transformed = transform(img)
                inputs = img_transformed.unsqueeze_(0)
                inputs = inputs.to(device)
                mhi_image = torch.cat((mhi_image, inputs), 1)
                if i == log:
                    inputs_image = mhi_image
                    mhi_image = torch.zeros_like(reset_tensor)
                    mhi_image = (torch.tensor(mhi_image)).to(device)

            #print(type(inputs_image))
            #print(inputs_image.shape)

            pred_idx = temporal_model(inputs_image).max(1)[1]
            y_pred_multi.append(pred_idx[0].data.item())

print(confusion_matrix(y_true_multi, y_pred_multi))
print(classification_report(y_true_multi, y_pred_multi))

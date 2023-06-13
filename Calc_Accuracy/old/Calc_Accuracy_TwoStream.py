from string import digits
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import glob
import os
import sys
import cv2
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
from Network.Temporal.ResNet import resnet50_temporal
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


Data_num = 300
log = 15

Spatial_model_path = r'E:\Research\TwoStreamCNN_2nd-Season\max_Val_acc_ResNet50_RGB_KTH_RGB_Split.pth'
Spatial_folder_path = r"E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\choise300\RGB"

Temporal_model_path = r'E:\Research\TwoStreamCNN_2nd-Season\max_Val_acc_ResNet50_MHI_KTH.pth'
Temporal_folder_path = r"E:\Research\TwoStreamCNN_2nd-Season\DataSet\KTH\Accuracy\choise300\MHI"

Class = os.listdir(Spatial_folder_path)
print(Class)

Spatial_true_multi = Temporal_true_multi = Twostream_true_multi_sum = Twostream_true_multi_average = [0]*Data_num + [1]*Data_num + [2]*Data_num + [3]*Data_num + [4]*Data_num + [5]*Data_num


def print_cmx(y_true, y_pred,name):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize = (10,7))
    sns.heatmap(df_cmx, annot=True)
    plt.savefig(str(name)) 


def Spatial_Accuracy():
    Spatial_pred_multi = []
    Spatial_result = []
    Spatial_missmatch_resulet = []
    Spatial_missmatch_num = []
    Spatial_model = resnet50_spatial().to(device)
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
            #output_spatial = F.softmax(Spatial_model(testImageTensor))         #softmax Ver
            output_spatial = Spatial_model(testImageTensor)
            pred_idx_spatial = output_spatial.max(1)[1]
            #if x != pred_idx_spatial:
                #Spatial_missmatch_resulet.append(Class[pred_idx_spatial])
                #cv2.imwrite((Class[pred_idx_spatial]) + '_' + str(y) + '.jpg',files)
            Spatial_missmatch_resulet.append(Class[pred_idx_spatial])
            Spatial_result.append(output_spatial.data[0])
            Spatial_pred_multi.append(pred_idx_spatial[0].data.item())
    return Spatial_pred_multi, Spatial_result, Spatial_missmatch_resulet


def Temporal_Accuracy():
    Temporal_result = []
    Temporal_pred_multi = []
    Temporal_missmatch_result = []
    Temporal_missmatch_num = []
    reset_tensor = []
    mhi_image = []
    mhi_image = (torch.tensor(mhi_image)).to(device)
    reset_tensor = (torch.tensor(reset_tensor)).to(device)
    Temporal_model = resnet50_temporal().to(device)
    Temporal_model.load_state_dict(torch.load(Temporal_model_path))
    Temporal_model.eval()
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    Base_files = glob.glob(Temporal_folder_path+"\*")
    for i, file in enumerate(Base_files):
        relay_file = glob.glob(file+"\*")
        for file in relay_file:
            image_file = glob.glob(file+"\*")
            # print(file)
            for i, files in enumerate(image_file):
                img = cv2.imread(files, cv2.IMREAD_GRAYSCALE)
                img = Image.fromarray(img)
                img_transformed = transform(img)
                inputs = img_transformed.unsqueeze_(0)
                inputs = inputs.to(device)
                mhi_image = torch.cat((mhi_image, inputs), 1)
                if i == (log-1):
                    inputs_image = mhi_image
                    mhi_image = torch.zeros_like(reset_tensor)
                    mhi_image = (torch.tensor(mhi_image)).to(device)
            #output_temporal = F.softmax(Temporal_model(inputs_image))      #softmax ver
            output_temporal = Temporal_model(inputs_image)
            pred_idx_temporal = output_temporal.max(1)[1]
            if i != pred_idx_temporal:
                Temporal_missmatch_result.append(files)
                Temporal_missmatch_num.append(Class[pred_idx_temporal])

            Temporal_result.append(output_temporal.data[0])
            Temporal_pred_multi.append(pred_idx_temporal[0].data.item())
    return Temporal_pred_multi, Temporal_result, Temporal_missmatch_result,Temporal_missmatch_num


def Twostream_Accuracy_sum(Spatial_result, Temporal_result):
    print(len(Spatial_result))
    print(len(Temporal_result))
    Twostream_pred_multi = []
    TwoStream_result = []
    for i in range(Data_num*(len(Class))):
        Twostream_result = Spatial_result[i]+Temporal_result[i]
        pred_idx_twostream = Twostream_result.max(0)[1]
        Twostream_pred_multi.append(pred_idx_twostream.data.item())
        TwoStream_result.append(Class[pred_idx_twostream])
        #print(i)
        #print("Twostream:"+str(Twostream_result))
        #print("Spatial  :" + str(Spatial_result[i]))
        #print("Temporal :" + str(Temporal_result[i]))
        #print("--------------------------------------------------------------------")
    return Twostream_pred_multi,TwoStream_result


def Twostream_Accuracy_average(Spatial_result, Temporal_result):
    Twostream_pred_multi = []
    TwoStream_result = []
    for i in range(Data_num*(len(Class))):
        Twostream_result = Spatial_result[i]+Temporal_result[i]
        Twostream_result = Twostream_result/2
        pred_idx_twostream = Twostream_result.max(0)[1]
        Twostream_pred_multi.append(pred_idx_twostream.data.item())
        TwoStream_result.append(Class[pred_idx_twostream])
        #print("Twostream:"+str(Twostream_result))
        #print("Spatial  :" + str(Spatial_result[i]))
        #print("Temporal :" + str(Temporal_result[i]))
        #print("--------------------------------------------------------------------")
    return Twostream_pred_multi,TwoStream_result




if __name__ == '__main__':
    Spatial_pred_multi, Spatial_result, Spatial_missmatch_resulet = Spatial_Accuracy()
    Temporal_pred_multi, Temporal_result, Temporal_missmatch_result,Temporal_missmatch_num = Temporal_Accuracy()
    Twostream_pred_multi_sum,TwoStream_result = Twostream_Accuracy_sum(Spatial_result, Temporal_result)
    #Twostream_pred_multi_average,TwoStream_result = Twostream_Accuracy_average(Spatial_result, Temporal_result)

    print("Spatial_confusion_matrix")
    print(confusion_matrix(Spatial_true_multi, Spatial_pred_multi))
    print(classification_report(Spatial_true_multi, Spatial_pred_multi,digits=4))
    print_cmx(Spatial_true_multi, Spatial_pred_multi,"Spatial")
    print("Temporal_confusion_matrix")
    print(confusion_matrix(Temporal_true_multi, Temporal_pred_multi))
    print(classification_report(Temporal_true_multi, Temporal_pred_multi,digits=4))
    print_cmx(Temporal_true_multi,Temporal_pred_multi,"Temporal")
    print("Twostream_confusion_matrix_sum")
    print(confusion_matrix(Twostream_true_multi_sum, Twostream_pred_multi_sum))
    print(classification_report(Twostream_true_multi_sum, Twostream_pred_multi_sum,digits=4))
    print_cmx(Temporal_true_multi,Temporal_pred_multi,"TwoStream")
    #print("Twostream_confusion_matrix_average")
    #print(confusion_matrix(Twostream_true_multi_average, Twostream_pred_multi_average))
    #print(classification_report(Twostream_true_multi_average, Twostream_pred_multi_average,digits=4))

import imp
from msilib.schema import Class
from pyexpat import features
from statistics import mode
import torch.nn as nn
import warnings
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import openpyxl
import glob
import pprint as pp
import cv2
import numpy as np
import csv
import torch
import shutil
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from Dataloder.Yuda_MotionHistory_dataloder import MotionHistory_Dataset
from torch.utils.data.dataset import Subset
from sklearn.model_selection import StratifiedKFold
from torchvision import models
from PIL import Image
from torchcam.methods import GradCAMpp
from itertools import chain
from torchcam.utils import overlay_mask
from torchvision import models
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image

modelPath = r"E:\Research\TwoStreamCNN_2nd-Season\Models\max_Val_acc_MobileNet_MHI_Wild-Life4th_part1\Fold-0_max_Val_acc_MobileNet_MHI_Wild-Life4th_part1.pth"
dataset = r"E:\Research\DataSet\Wild_Life\4th_Season\Image\train\MHI"
modelName = r"Fold-0_max_Val_acc_MobileNet_Co-Occurrence_Wild-Life4th_part1"

transform = transforms.Compose([
	transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
	transforms.ToTensor(),
])

class MobileNet():
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(5,32,3)
        self.transfromClassifier()
        self.transGrad(True)
    def transfromClassifier(self):
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1280, 3)
        )
    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo

class Cam():
	def __init__(self, loadmodel,target_layer):
		self.model = loadmodel
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		self.cam_extractor = GradCAMpp(self.model.eval(),target_layer)
	def GradCam(self, image, label):
		output = self.model((image.unsqueeze(0)).to(self.device))
		pred = (F.softmax((self.model((image.unsqueeze(dim=0)).to(self.device)))))
		pred_idx = pred.max(1)[1]
		cam = self.cam_extractor(output.squeeze(0).argmax().item(), output)
		result = to_pil_image(cam[0], mode="F")
		pred_value = pred.data[0].tolist()
		return result, pred_idx, pred_value
	def __call__(self, image, label):
		result_img, pred_idx, pred_value = self.GradCam(image, label)

if __name__ == '__main__':
	Mobilenet = MobileNet()
	model = Mobilenet.model.cuda()
	model.load_state_dict(torch.load(modelPath, map_location='cuda:0'))
	ClassList = os.listdir(dataset)
	Data = MotionHistory_Dataset(dataset, transform, 5)
	Gradcam = Cam(loadmodel=model,target_layer=model.features[0])
	print(Gradcam)
	idx = 0
	#os.mkdir("GradTest")
	print(len(Data))
	for images, labels in Data:
		Gradcam_img, pred_idx, pred_value = Gradcam.GradCam(image=images, label=labels)
		Gradcam_img = Gradcam_img.convert('RGB')
		Gradcam_img.save(str(idx)+".jpg")
		exit()
		idx+=1
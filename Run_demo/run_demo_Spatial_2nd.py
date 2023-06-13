import os
import sys
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
sys.path.append('../')
from PIL import Image
from torchvision import models
from Network.Spatial.MobileNet import MobileNet_V2
from fps import DispFps

class MobileNet_V2_Spatial():
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[0] = nn.Dropout(0.5)
        self.model.classifier[1] = nn.Linear(1280, 3)
        self.transGrad(True)
    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo

class BaseTransform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
        ])
    def __call__(self, img):
        return self.base_transform(img)

class Run_Spatial():
    def __init__(self):
        self.classList = ['Boar','Bear','Others']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = BaseTransform()

    def cv2pil(self,image_cv):
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_cv)
        image_pil = image_pil.convert('RGB')
        return image_pil

    def run_demo(self, model_path, movie_path):
        MobileNet = MobileNet_V2()
        Spatial_model = MobileNet.model.cuda()
        Spatial_model.load_state_dict(torch.load(model_path))
        Spatial_model.eval()
        capture= cv2.VideoCapture(movie_path)
        ret,frame = capture.read()
        dispFps = DispFps()                                                                                  
        while(capture.isOpened()):
            ret,frame = capture.read() 
            PILframe = self.cv2pil(frame)
            resize_img  =  self.transform(PILframe)
            output_spatial = F.softmax(Spatial_model((resize_img.unsqueeze(dim=0)).to(self.device))) 
            pred_idx_spatial = output_spatial.max(1)[1]
            batch_probs, batch_indices = output_spatial.sort(dim=1, descending=True)
            cv2.resize(frame,(600,600))
            cv2.putText(frame,str(self.classList[pred_idx_spatial[0].data.item()]), (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
            for probs,indices in zip(batch_probs,batch_indices):
                for k in range(3):
                    cv2.putText(frame,str((f"Top-{k + 1} {probs[k]:.2%} {self.classList[indices[k]]}")), (0, 100+(50*k)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)
            dispFps.disp(frame)
            cv2.imshow("Spatial",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                capture.release()
                cv2.destroyAllWindows()
                break

    def __call__(self, model_path=None, movie_path=0):
        self.run_demo(model_path, movie_path)

test = Run_Spatial()
test(model_path=r"I:\TwoStreamCNN_2ndSeason\G_Model\max_Val_acc_MoileNet_RGB_20BN-6Class_RGB_full_part1.pth",
    movie_path=r"I:\TwoStreamCNN_2ndSeason\Run_demo\Boar (3).mp4")
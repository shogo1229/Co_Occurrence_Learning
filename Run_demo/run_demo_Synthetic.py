from itertools import count
import os
import sys
import glob
import numpy as np
import cv2
import torch
import time
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
sys.path.append('../')
from torchvision import models
from PIL import Image
from timeit import default_timer as timer
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

class Run_Synthetic():
    def __init__(self,Threshold = 10,Pixel_Max_Value=255,Pixel_Min_Value=0,Tau=5):
        self.classList = ['Boar','Bear','Others']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = BaseTransform()
        self.THRESHOLD = Threshold
        self.TAU = Tau
        self.PIXEL_MAX_VALUE = Pixel_Max_Value
        self.PIXEL_MIN_VALUE = Pixel_Min_Value
        self.kernel = np.ones((5,5),np.uint8)

    def cv2pil(self,image_cv):
        image_pil=Image.fromarray(image_cv)
        return image_pil

    def createMHI(self,MHI,frameDifferenceImage,motionHistoryImage):
        idx_PIXEL_MAX_VALUE = np.where(frameDifferenceImage == self.PIXEL_MAX_VALUE)                    #更新された部分のインデックスを取得(返り値はタプル)
        MHI[idx_PIXEL_MAX_VALUE] = self.TAU                                                             #更新された部分の値にTAU(残すフレーム数)を代入
        idx_TAU = np.where(MHI > self.PIXEL_MIN_VALUE)                                                  #画素値が1以上の部分のインデックスを取得
        MHI[idx_TAU] = MHI[idx_TAU] - 1                                                                 #画素値が1以上の全て画素値に対して-1
        MHI[MHI<self.PIXEL_MIN_VALUE] = self.PIXEL_MIN_VALUE                                            #画素値が-1になった部分を全て0にする
        motionHistoryImage = MHI/self.TAU*self.PIXEL_MAX_VALUE                                          #画素値が1ずつしか違わないと分かりづらいから変換
        motionHistoryImage = np.fix(motionHistoryImage)                                                 #一個上でfloat型になったから整数に
        motionHistoryImage = motionHistoryImage.astype(np.uint8)                                        #floatで返すと0,255の値で返されるからunit8型に変換
        return motionHistoryImage,MHI

    def run_demo(self, model_path, movie_path):
        MobileNet = MobileNet_V2()
        Synthetic_model = MobileNet.model.cuda()
        Synthetic_model.load_state_dict(torch.load(model_path))
        Synthetic_model.eval()
        capture = cv2.VideoCapture(movie_path) 
        ret,frameImage = capture.read()                                                                 #フレーム読み込み
        MHI = np.zeros((frameImage.shape[0], frameImage.shape[1]), np.uint8)
        motionHistoryImage = np.zeros((frameImage.shape[0], frameImage.shape[1], 1), np.uint8)
        prevImage = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                         #frameImageをグレースケール化し,t-2の画像とする
        ret,frameImage = capture.read()                                                                 #フレーム読み込み
        prevImage2 = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                        #frameImageをグレースケール化し,t-1の画像とする
        ret,frameImage = capture.read()                                                                 #フレーム読み込み
        currentImage = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                      #frameImageをグレースケール化し,tの画像とする
        dispFps = DispFps()
        while(capture.isOpened()):
            ret,frameImage = capture.read()                                                             #フレーム読み込み
            nextImage = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                 #frameImageをグレースケール化し,t+1の画像とする
            ret,frameImage = capture.read()                                                         #フレーム読み込み                                                       
            nextImage2 = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                #frameImageをグレースケール化し,t+2の画像とする
            differenceImage1 = cv2.absdiff(prevImage2,prevImage)                                    #(t-1)-(t-2)の差分
            differenceImage2 = cv2.absdiff(prevImage,currentImage)                                  #(t-1)-tの差分
            differenceImage3 = cv2.absdiff(currentImage,nextImage)                                  #t-(t+1)の差分} - tの差分
            differenceImage4 = cv2.absdiff(nextImage,nextImage2)                                    #(t+1)-(t+2)の差分
            ret,differenceImage1 = cv2.threshold(differenceImage1,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            ret,differenceImage2 = cv2.threshold(differenceImage2,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            ret,differenceImage3 = cv2.threshold(differenceImage3,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            ret,differenceImage4 = cv2.threshold(differenceImage4,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            frameDifferenceImage1 = cv2.bitwise_and(differenceImage2,differenceImage1,mask=differenceImage2)                  #差分画像同士のAnd演算
            frameDifferenceImage2 = cv2.bitwise_and(differenceImage4,differenceImage3,mask=differenceImage4)                  #差分画像同士のAnd演算
            frameDifferenceImage = cv2.bitwise_or(frameDifferenceImage1,frameDifferenceImage2,mask=frameDifferenceImage1)    #差分画像同士のAnd演算
            frameDifferenceImage = cv2.medianBlur(frameDifferenceImage,3)                                                     #メディアンフィルタでノイズ除去
            frameDifferenceImage = cv2.dilate(frameDifferenceImage,kernel=self.kernel,iterations=1)                           #オープニング処理
            frameDifferenceImage = cv2.erode(frameDifferenceImage,kernel=self.kernel,iterations=1)                            #クロージング処理
            motionHistoryImage,MHI = self.createMHI(MHI,frameDifferenceImage,motionHistoryImage)
            prevImage = prevImage2.copy()
            prevImage2 = currentImage.copy()
            currentImage = nextImage.copy()
            nextImage = nextImage2.copy()
            motionHistoryImage = cv2.cvtColor(motionHistoryImage,cv2.COLOR_GRAY2BGR)
            Synthetic_Image = cv2.addWeighted(src1=frameImage,alpha=1,src2=motionHistoryImage,beta=0.5,gamma=0)
            PILframe = self.cv2pil(Synthetic_Image)
            resize_img  =  self.transform(PILframe)
            output_spatial = F.softmax(Synthetic_model((resize_img.unsqueeze(dim=0)).to(self.device))) 
            pred_idx_spatial = output_spatial.max(1)[1]
            batch_probs, batch_indices = output_spatial.sort(dim=1, descending=True)
            cv2.resize(Synthetic_Image,(600,600))
            cv2.putText(Synthetic_Image,str(self.classList[pred_idx_spatial[0].data.item()]), (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
            for probs,indices in zip(batch_probs,batch_indices):
                for k in range(3):
                    cv2.putText(Synthetic_Image,str((f"Top-{k + 1} {probs[k]:.2%} {self.classList[indices[k]]}")), (0, 100+(50*k)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)
            dispFps.disp(Synthetic_Image)
            cv2.imshow("Synthetic",Synthetic_Image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                capture.release()
                cv2.destroyAllWindows()
                break
    def __call__(self, model_path=None, movie_path=0):
        self.run_demo(model_path, movie_path)

test = Run_Synthetic()
test(model_path=r"I:\TwoStreamCNN_2ndSeason\G_Model\max_Val_acc_MobileNet_20BN-6class_Synthec_log5_part1.pth",
        movie_path=r"I:\TwoStreamCNN_2ndSeason\Run_demo\Boar (3).mp4")
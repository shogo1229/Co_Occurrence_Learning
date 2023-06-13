import warnings
warnings.simplefilter('ignore', Warning)
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset                  
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
from torch.utils.data.dataset import Subset
from Network.Spatial.ResNet import resnet50_spatial
from Network.Spatial.VGG16 import VGG16_Spatial
from Network.Spatial.GhostNet import Yuda_GhostNet
from Network.Spatial.MobileNet import MobileNet_V2_Spatial, MobileNet_V3_small_Spatial,MobileNet_V3_large_Spatial
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedKFold

#-----------------------------ハイパラ等の宣言-----------------------------#
EpochNum = 100                                          
Height = 224                                            
Width = 224                                           
BatchSize = 16
Fold = 1
LearningRate = 0.001 
#----------------------データセット、モデルパスの宣言-------------------------#

DatasetPath = r""  # データセットのパス
valDatasetPath = r""  # 検証データセットのパス
modelPath = r""  # モデルの保存先パス
ResumeModel = r"" #追加学習を行うモデルのパス
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 使用するデバイスの指定 (GPUが利用可能ならばGPUを使用)


#----------------------学習用コード-------------------------#


class Trainer():
    def __init__(self, model, optimizer, criterion, trainLoader, valLoader, transform, epoch):
        self.model = model  # モデル
        self.ResumeEpoch = epoch  # レジュームするエポック
        self.optimizer = optimizer  # オプティマイザ
        self.criterion = criterion  # 損失関数
        self.trainLoader = trainLoader  # 訓練データのデータローダー
        self.valLoader = valLoader  # 検証データのデータローダー
        self.transform = transform  # 画像の前処理
        self.totalTrainLoss = []  # 各エポックの訓練ロスを保存するリスト
        self.TrainCorrect = []  # 各エポックの訓練データの正解率を保存するリスト
        self.totalValLoss = []  # 各エポックの検証ロスを保存するリスト
        self.ValCorrect = []  # 各エポックの検証データの正解率を保存するリスト
        self.Fig = plt.figure(figsize=[10, 10])  # グラフのフィギュア

    def Train(self, epoch):
        self.model.train()  # モデルを訓練モードに設定
        train_loss, train_acc = 0.0, 0.0
        t_loss, t_acc =  0.0, 0.0
        train_log = ""
        for batchIdx, (img, label) in enumerate(self.trainLoader):
            img, label = Variable(img.cuda()), Variable(label.cuda())
            output = self.model(img)
            loss = self.criterion(output, label)
            train_loss += loss.data.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pred = output.data.max(dim=1)[1]
            train_acc += pred.eq(label.data).cpu().sum()
            t_loss = train_loss / ((batchIdx + 1) * BatchSize)
            t_acc  = 100 * train_acc.data.item() / ((batchIdx + 1) * BatchSize)
            train_log = "epoch : {:3} train_loss : {:3.12} train_acc : {:3.12}".format(str(epoch + 1), str(t_loss), str(t_acc))
            print("\r"+train_log,end="")
        self.totalTrainLoss.append(t_loss)  # エポックごとの訓練ロスを保存
        self.TrainCorrect.append(t_acc)  # エポックごとの訓練データの正解率を保存

        self.model.eval()  # モデルを評価モードに設定
        val_loss, val_acc, min_acc = 0.0, 0.0, 0.0
        v_loss, v_acc = 0.0, 0.0
        val_log = ""
        with torch.no_grad():
            for batchIdx, (img, label) in enumerate(self.valLoader):
                img, label = Variable(img.cuda()), Variable(label.cuda())
                output = self.model(img)
                loss = self.criterion(output, label)
                val_loss += loss.data.item()
                pred = output.data.max(dim=1)[1]
                val_acc += pred.eq(label.data).cpu().sum()
                v_loss = val_loss / ((batchIdx + 1) * BatchSize)
                v_acc = 100 * val_acc.data.item() / ((batchIdx + 1) * BatchSize)
                val_log = train_log + " val_loss : {:3.9} val_acc : {:3.9}".format(str(v_loss), str(v_acc))
                print("\r"+val_log,end="")
            self.totalValLoss.append(v_loss)  # エポックごとの検証ロスを保存
            self.ValCorrect.append(v_acc)  # エポックごとの検証データの正解率を保存
        print()
        if v_acc > min_acc:
            self.max_acc = v_acc
            with open("models/"+str(modelPath)+"/"+str(modelPath)+".pth", "wb") as savePath:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}, savePath)

    def graph_plot(self):
        plt.clf()
        plt.style.use('ggplot')
        lossFig = self.Fig.add_subplot(2, 1, 1) 
        plt.title('Loss Graph')
        plt.plot(self.totalTrainLoss, label='train loss')  # 訓練ロスをプロット
        plt.plot(self.totalValLoss, label='validation loss')  # 検証ロスをプロット
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        accuracyFig = self.Fig.add_subplot(2, 1, 2) 
        plt.title('Accuracy Graph (max_Val-acc :'+str(round(float(self.max_acc),4))+'%)')
        plt.minorticks_on()
        plt.ylim(0, 100)
        plt.plot(self.TrainCorrect, label='train acc')  # 訓練データの正解率をプロット
        plt.plot(self.ValCorrect, label='validation acc')  # 検証データの正解率をプロット
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    def main(self):
        for epoch in range(self.ResumeEpoch, EpochNum):
            self.Train(epoch)
            self.graph_plot()
            self.Fig.savefig("models/"+str(modelPath)+"/"+str(modelPath)+".png")

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((Height, Width), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
        ])
    ss = ShuffleSplit(n_splits=Fold, test_size=0.2, random_state=0)
    dataset_labels = []
    dataset = datasets.ImageFolder(root=DatasetPath, transform=transform)
    print("Load DataSet 「"+str(DatasetPath)+"」")
    MobileNet = MobileNet_V2_Spatial()
    model = MobileNet.model.cuda()
    os.mkdir("models/"+str(modelPath))
    for _fold, (train_index, val_index) in enumerate(ss.split(dataset)):  
        train_dataset = Subset(dataset, train_index)
        val_dataset = Subset(dataset, val_index)          
        trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
        valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=BatchSize, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)  # シャッフルk分割
        optimizer = optim.Adam(model.parameters(), lr=LearningRate)
        checkpoint = torch.load(ResumeModel)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']    
        criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        train_dataset = Subset(dataset, train_dataset)
        print("Learning rate:", LearningRate)
        print("ResumeEpoch:", epoch)
        train = Trainer(model, optimizer, criterion, trainLoader, valLoader, transform, epoch)
        train.main()

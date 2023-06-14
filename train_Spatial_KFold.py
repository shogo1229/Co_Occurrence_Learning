from ast import Pass  # astモジュールからPassクラスをインポート
import warnings  # warningsモジュールをインポート
warnings.simplefilter('ignore', Warning)  # 警告を無視するように設定
import torch  # torchモジュールをインポート
import torch.nn as nn  # torch.nnモジュールからnnクラスをインポート
import torch.optim as optim  # torch.optimモジュールからoptimクラスをインポート
import torchvision.transforms as transforms  # torchvision.transformsモジュールをインポート
import torch.nn.functional as F  # torch.nn.functionalモジュールをインポート
import os  # osモジュールをインポート
import pprint as pp  # pprintモジュールをインポート
import numpy as np  # numpyモジュールをインポート
import matplotlib.pyplot as plt  # matplotlib.pyplotモジュールをインポート
from torch.utils.data import Dataset  # torch.utils.dataモジュールからDatasetクラスをインポート
from torchvision import datasets, transforms  # torchvisionモジュールからdatasetsクラス、transformsモジュールをインポート
from torch.autograd import Variable  # torch.autogradモジュールからVariableクラスをインポート
from PIL import Image  # PILモジュールからImageクラスをインポート
from statistics import mean  # statisticsモジュールからmean関数をインポート
from Network.Spatial.ResNet import resnet50_spatial  # Network.Spatial.ResNetモジュールからresnet50_spatialクラスをインポート
from Network.Spatial.VGG16 import VGG16_Spatial  # Network.Spatial.VGG16モジュールからVGG16_Spatialクラスをインポート
from Network.Spatial.GhostNet import Yuda_GhostNet  # Network.Spatial.GhostNetモジュールからYuda_GhostNetクラスをインポート
from Network.Spatial.MobileNet import MobileNet_V2_Spatial, MobileNet_V3_small_Spatial, MobileNet_V3_large_Spatial  # Network.Spatial.MobileNetモジュールからMobileNet_V2_Spatialクラス、MobileNet_V3_small_Spatialクラス、MobileNet_V3_large_Spatialクラスをインポート
from torch.utils.data import Dataset, DataLoader  # torch.utils.dataモジュールからDatasetクラス、DataLoaderクラスをインポート
from torch.utils.data.dataset import Subset  # torch.utils.data.datasetモジュールからSubsetクラスをインポート
from sklearn.model_selection import ShuffleSplit, StratifiedKFold  # sklearn.model_selectionモジュールからShuffleSplitクラス、StratifiedKFoldクラスをインポート

EpochNum = 15  # エポック数
Height = 224  # 画像の高さ
Width = 224  # 画像の幅
BatchSize = 64  # バッチサイズ
Fold = 5  # 交差検証の分割数
Accuracy = [0]*Fold  # 各分割ごとの精度を格納するリスト
DatasetPath = r""  # データセットのパス
modelPath = r""  # モデルのパス
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # CUDAが利用可能ならGPUを使用して計算する

class Trainer():
    def __init__(self,model, optimizer, criterion, trainLoader, valLoader, transform,_fold):
        self.model = model  # モデル
        self.optimizer = optimizer  # オプティマイザ
        self.criterion = criterion  # 損失関数
        self.trainLoader = trainLoader  # 訓練データのデータローダー
        self.valLoader = valLoader  # 検証データのデータローダー
        self.transform = transform  # 画像の前処理
        self.fold = _fold  # 分割数
        self.max_acc = 0  # 最大精度
        self.totalTrainLoss = []  # 訓練データの損失の履歴
        self.TrainCorrect = []  # 訓練データの正解率の履歴
        self.totalValLoss = []  # 検証データの損失の履歴
        self.ValCorrect = []  # 検証データの正解率の履歴
        self.Fig = plt.figure(figsize=[10,10])  # グラフの描画領域の初期化

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
            t_loss = train_loss/((batchIdx+1)*BatchSize)
            t_acc  = 100*train_acc.data.item() / ((batchIdx+1)*BatchSize)
            train_log = "epoch : {:3} train_loss : {:3.12} train_acc : {:3.12}".format(str(epoch+1), str(t_loss), str(t_acc))
            print("\r"+train_log,end="")
        self.totalTrainLoss.append(t_loss)
        self.TrainCorrect.append(t_acc)

        self.model.eval()  # モデルを評価モードに設定
        val_loss, val_acc = 0.0, 0.0
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
                v_loss = val_loss/((batchIdx+1)*BatchSize)
                v_acc = 100*val_acc.data.item() / ((batchIdx+1)*BatchSize)
                val_log = train_log + " val_loss : {:3.9} val_acc : {:3.9}".format(str(v_loss), str(v_acc))
                print("\r"+val_log,end="")
            self.totalValLoss.append(v_loss)
            self.ValCorrect.append(v_acc)
        print()
        if v_acc > self.max_acc:
            self.max_acc = v_acc
            with open("models/"+str(modelPath)+"/Fold-"+str(self.fold)+"_"+str(modelPath)+".pth", "wb") as savePath:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss}, savePath)

    def graph_plot(self):
        plt.clf()
        plt.style.use('ggplot')
        lossFig = self.Fig.add_subplot(2,1,1)
        plt.title('Loss Graph')
        plt.plot(self.totalTrainLoss, label='train loss')
        plt.plot(self.totalValLoss, label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        accuracyFig = self.Fig.add_subplot(2,1,2)
        plt.title('Accuracy Graph (max_Val-acc :'+str(round(float(self.max_acc),4))+'%)')
        plt.minorticks_on()
        plt.ylim(0,100)
        plt.plot(self.TrainCorrect, label='train acc')
        plt.plot(self.ValCorrect, label='validation acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    def main(self):
        for epoch in range(EpochNum):
            self.Train(epoch)
            self.graph_plot()
            self.Fig.savefig("models/"+str(modelPath)+"/Fold-"+str(self.fold)+"_"+str(modelPath)+".png")
        return self.max_acc

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((Height, Width), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5])
		])
    skf = StratifiedKFold(n_splits=Fold)  # 分割数を指定して層化K分割法のインスタンスを作成
    ss = ShuffleSplit(n_splits=Fold, test_size=0.2, random_state=0)  # 分割数とテストデータの割合を指定してシャッフルK分割法のインスタンスを作成
    os.mkdir("Models/"+str(modelPath))  # モデルの保存先ディレクトリを作成
    criterion = nn.CrossEntropyLoss()  # 損失関数としてCrossEntropyLossを使用
    dataset = datasets.ImageFolder(root=DatasetPath, transform=transform)  # データセットを読み込み、前処理を適用
    print("Load DataSet 「"+str(DatasetPath)+"」")

    for _fold, (train_index, val_index) in enumerate(skf.split(dataset.imgs, dataset.targets)):
        train_data = Subset(dataset, train_index)  # 訓練データのサブセットを作成
        val_data = Subset(dataset, val_index)  # 検証データのサブセットを作成
        print("Fold : "+str(_fold+1)+"========================================================")
        trainLoader = DataLoader(train_data, batch_size=BatchSize, shuffle=True, num_workers=4)  # 訓練データのデータローダーを作成
        valLoader = DataLoader(val_data, batch_size=BatchSize, shuffle=True, num_workers=4)  # 検証データのデータローダーを作成

        model = MobileNet_V2_Spatial().to(device)  # MobileNetV2モデルを使用
        # モデルの重みを保存しておく
        with open("models/"+str(modelPath)+"/Fold-"+str(_fold+1)+"_"+str(modelPath)+"_init.pth", "wb") as savePath:
            torch.save(model.state_dict(), savePath)

        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)  # SGDを使用
        trainer = Trainer(model, optimizer, criterion, trainLoader, valLoader, transform, _fold+1)  # トレーナーを作成
        accuracy[_fold] = trainer.main()

    print("Max Accuracy : ", max(accuracy))  # 最大精度を表示

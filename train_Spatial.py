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

EpochNum = 21                                          
Height = 224                                            
Width = 224                                           
BatchSize = 16
LearningRate = 0.001                           
DatasetPath = r"F:\AROB_Journal\Normal_MHI_log5\train\RGB" 
valDatasetPath =r"F:\AROB_Journal\Normal_MHI_log5\val\RGB"
modelPath = r"MobileNet_RGB_AROB-Journal"          
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")       



class Trainer():
    def __init__(self,model, optimizer, criterion, trainLoader, valLoader, transform):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainLoader = trainLoader
        self.valLoader = valLoader
        self.transform = transform
        self.totalTrainLoss = []
        self.TrainCorrect = []
        self.totalValLoss = []
        self.ValCorrect = []
        self.Fig = plt.figure(figsize=[10,10])
        self.max_acc = 0.0
        self.min_loss = 100
    def Train(self,epoch):
        self.model.train()
        train_loss,train_acc = 0.0,0.0
        t_loss,t_acc =  0.0,0.0
        train_log = ""
        for batchIdx,(img,label) in enumerate(self.trainLoader):
            img,label = Variable(img.cuda()),Variable(label.cuda())
            output = self.model(img)
            loss = self.criterion(output,label)
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
        self.model.eval()
        val_loss,val_acc = 0.0,0.0
        v_loss,v_acc = 0.0,0.0
        val_log = ""
        with torch.no_grad():
            for batchIdx,(img,label) in enumerate(self.valLoader):
                img,label = Variable(img.cuda()),Variable(label.cuda())
                output = self.model(img)
                loss = self.criterion(output,label)
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
            with open("models/"+str(modelPath)+"/"+str(modelPath)+"_max-Acc.pth", "wb") as savePath:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,},savePath)
        if v_loss < self.min_loss:
            self.min_loss = v_loss
            with open("models/"+str(modelPath)+"/"+str(modelPath)+"_min-Loss.pth", "wb") as savePath:
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,},savePath)

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
            self.Fig.savefig("models/"+str(modelPath)+"/"+str(modelPath)+".png")

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((Height, Width), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
			mean=[0.5, 0.5, 0.5],
			std=[0.5, 0.5, 0.5])
		])
    
    os.mkdir("Models/"+str(modelPath))
    criterion = nn.CrossEntropyLoss()
    train_dataset = datasets.ImageFolder(root=DatasetPath, transform=transform)
    val_dataset = datasets.ImageFolder(root=valDatasetPath, transform=transform)
    print("Load DataSet 「"+str(DatasetPath)+"」")
    #train_dataset, val_dataset, train_label, test_label = train_test_split(dataset, dataset_labels, test_size=0.2, random_state=100,shuffle=True
    trainLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    valLoader = torch.utils.data.DataLoader(val_dataset, batch_size=BatchSize, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    print(len(train_dataset),len(val_dataset))
    MobileNet = MobileNet_V2_Spatial()
    model = MobileNet.model.cuda()
    #os.mkdir("Models/"+str(modelPath))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=LearningRate)
    #train_dataset = Subset(dataset,train_dataset)
    print("Learning rate :",LearningRate)
    train = Trainer(model, optimizer, criterion,trainLoader, valLoader, transform)
    train.main()
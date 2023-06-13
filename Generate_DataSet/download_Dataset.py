import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

if __name__ == '__main__':
    datadir = r"E:\Research\TwoStreamCNN_2nd-Season\DataSet\MNIST"
    trainset = datasets.hmdb51(datadir, download=True)

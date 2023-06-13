import torch.nn as nn
from torchvision import models


class MobileNet_V2_Spatial():
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[0] = nn.Dropout(0.5)
        self.model.classifier[1] = nn.Linear(1280, 6)
        self.transGrad(True)
    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo

class MobileNet_V3_small_Spatial():
    def __init__(self):
        self.model = models.mobilenet_v3_small(pretrained=False)
        #self.model.classifier[2] = nn.Dropout(0.5)
        self.model.classifier[3] = nn.Linear(1024, 3)
        self.transGrad(True)

    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo

class MobileNet_V3_large_Spatial():
    def __init__(self):
        self.model = models.mobilenet_v3_large(pretrained=True)
        #self.model.classifier[2] = nn.Dropout(0.5)
        self.model.classifier[3] = nn.Linear(1024, 3)
        self.transGrad(True)

    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo


import torch.nn as nn
from torchvision import models


class MobileNet_V2_Temporal():
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(5,32,3)
        self.transfromClassifier()
        self.transGrad(True)

    def transfromClassifier(self):
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1280, 6)
        )

    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo

class MobileNet_V3_small_Temporal():
    def __init__(self):
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(5,32,3)
        self.model.classifier[2] = nn.Dropout(0.5)
        self.model.classifier[3] = nn.Linear(1024, 3)
        self.transGrad(True)

    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo

class MobileNet_V3_large_Temporal():
    def __init__(self):
        self.model = models.mobilenet_v3_large(pretrained=True)
        self.model.features[0][0] = nn.Conv2d(5,32,3)
        #self.model.classifier[2] = nn.Dropout(0.5)
        self.model.classifier[3] = nn.Linear(1024, 3)
        self.transGrad(True)

    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo


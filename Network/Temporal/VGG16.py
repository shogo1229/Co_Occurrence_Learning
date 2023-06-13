import torch.nn as nn
from torchvision import models

class VGG16_Temporal():
    def __init__(self):
        self.model = models.vgg16_bn(pretrained=True)
        self.model.features[0] = nn.Conv2d(5, 64, 3)
        self.transformClassifier()
        self.transGrad(True)

    def transformClassifier(self):
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 3)  # クラス数
        )
    # 微分するかしないかみたいなやつ

    def transGrad(self, Boo):
        for p in self.model.features.parameters():
            p.requires_grad = Boo
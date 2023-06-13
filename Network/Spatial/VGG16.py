import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import *

class VGG16_Spatial():
	def __init__(self):
		self.model = models.vgg16_bn(pretrained=False)
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
			nn.Linear(4096, 11)              #クラス数
			)
	# 微分するかしないかみたいなやつ
	def transGrad(self, Boo):
		for p in self.model.features.parameters():
			p.requires_grad = Boo

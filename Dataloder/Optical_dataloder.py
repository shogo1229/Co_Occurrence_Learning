from types import LambdaType
import torch, os, glob
import torchvision.transforms as transforms
from PIL import Image
from itertools import chain
import collections

class MotionDataset(torch.utils.data.Dataset):
	def __init__(self, _imgPaths, _transform=None, _inChannels = 10, _imgHeight=224, _imgWidth=224):
		self.Transform = _transform
		self.ImagePaths = _imgPaths
		self.LabelPaths = None
		self.LabelsIndex = None
		self.Labels = None
		self.LabelName = None
		self.InChannels = _inChannels
		self.FlowLabel = []
		self.FlowPaths = []
		self.SumDataset()
		self.FlowLabel = list(chain.from_iterable(self.FlowLabel))
		self.FlowPaths = list(chain.from_iterable(self.FlowPaths))
		self.LabelPathsCounter = sorted(collections.Counter(self.FlowPaths).items())
		self.LabelCounter = sorted(collections.Counter(self.FlowLabel).items())
		self.Height = _imgHeight
		self.Width = _imgWidth

	def SumDataset(self):
		self.LabelPaths = glob.glob(self.ImagePaths+r"/u/**/")
		self.Labels = [f.replace(self.ImagePaths+r"/u\v_", "").split("_")[0] for f in self.LabelPaths]
		self.LabelName = list(set(self.Labels))
		self.LabelName.sort()
		self.LabelsIndex = [self.LabelName.index(f) for f in self.Labels]
		for i, l in enumerate(self.LabelPaths):
			flowSum = int(len(glob.glob(l+"/*"))/self.InChannels)
			self.FlowLabel.append([self.LabelsIndex[i]]*flowSum)
			self.FlowPaths.append([l+"/frame{}.jpg".format(str(idx).zfill(6))for idx in range(1, flowSum*self.InChannels+1)])


	def __getitem__(self, index):
		xImage = None
		yImage = None
		label = self.FlowLabel[index]
		flow = torch.FloatTensor(2*self.InChannels, self.Height, self.Width)
		for channel in range(self.InChannels):
			idx = index*self.InChannels+channel
			xImagePath = self.FlowPaths[idx]
			yImagePath = xImagePath.replace("/u", "/v")
			with open(xImagePath, "rb") as f:
				xImage = Image.open(f)
				xImage = xImage.convert('L')
			with open(yImagePath, "rb") as f:
				yImage = Image.open(f)
				yImage = yImage.convert('L')
			if self.Transform:
				xImage = self.Transform(xImage)
				yImage = self.Transform(yImage)
			flow[2*(channel),:,:] = xImage
			flow[2*(channel)+1,:,:] = yImage
		sample = (flow, label)
		return sample

	def __len__(self):
		return len(self.FlowLabel)

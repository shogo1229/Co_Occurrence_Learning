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
		self.LabelPaths = glob.glob(self.ImagePaths+r"/**")
		self.Labels = [f.replace(self.ImagePaths+r"\v_", "").split("_")[0] for f in self.LabelPaths]
		self.LabelName = list(set(self.Labels))
		self.LabelName.sort()
		self.LabelsIndex = [self.LabelName.index(f) for f in self.Labels]
		for i, l in enumerate(self.LabelPaths):
			flowSum = int(len(glob.glob(l+"/*"))/self.InChannels)
			self.FlowLabel.append([self.LabelsIndex[i]]*flowSum)
			self.FlowPaths.append([l+"/frame{}.jpg".format(str(idx).zfill(6))for idx in range(0, flowSum*self.InChannels)])

	def __getitem__(self, index):
		xImage = None
		label = self.FlowLabel[index]
		mhi = torch.FloatTensor(self.InChannels, self.Height, self.Width)
		for channel in range(self.InChannels):
			idx = index*self.InChannels+channel
			xImagePath = self.FlowPaths[idx]
			with open(xImagePath, "rb") as f:
				xImage = Image.open(f)
				xImage = xImage.convert('L')			#L グレースケール変換 RGB 8bit×3
			if self.Transform:
				xImage = self.Transform(xImage)
			mhi[(channel),:,:] = xImage
		sample = (mhi, label)
		return sample

	def __len__(self):
		return len(self.FlowLabel)

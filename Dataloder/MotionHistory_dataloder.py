from types import LambdaType
import torch, os, glob
import torchvision.transforms as transforms
from PIL import Image
from itertools import chain
import collections
import pprint as pp


class MotionHistory_Dataset(torch.utils.data.Dataset):
	def __init__(self, _imgPaths, _transform=None, _inChannels=5, _imgHeight=224, _imgWidth=224):
		self.Transform = _transform
		self.ImagePaths = _imgPaths
		self.LabelPaths = None
		self.LabelsIndex = None
		self.Labels = None
		self.LabelName = None
		self.InChannels = _inChannels
		self.step = _inChannels
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
		self.Labels = self.LabelPaths
		self.LabelName = list(set(self.Labels))
		self.LabelName.sort()
		self.LabelsIndex = [self.LabelName.index(f) for f in self.Labels]
		for idx, label in enumerate(self.LabelPaths):
			for i, l in enumerate(glob.glob(label+r"/*")):
				flowSum = int(len(glob.glob(l+"/*"))/self.InChannels)
				FList = glob.glob(l+"/*")
				self.FlowLabel.append([self.LabelsIndex[idx]]*(len(glob.glob(l+"/*"))-(self.step-1)))
				for g in range(len(glob.glob(l+"/*"))-(self.step-1)):
					L = []
					for x in range(g,g+self.step):
						L.append(FList[x])
					self.FlowPaths.append(L)
		#pp.pprint(self.FlowLabel)
		#pp.pprint(self.FlowPaths)
		#print(len(self.FlowLabel),len(self.FlowPaths))

	def __getitem__(self, index):
		MHI = None
		self.checkList =[]
		label = self.FlowLabel[index]
		mhi = torch.FloatTensor(self.InChannels, self.Height, self.Width)
		for channel in range(self.InChannels):
			idx = index*self.InChannels+channel
			xImagePath = self.FlowPaths[idx]
			with open(xImagePath, "rb") as f:
				MHI = Image.open(f)
				MHI = MHI.convert('L')			#L グレースケール変換 RGB 8bit×3
			if self.Transform:
				MHI = self.Transform(MHI)
			mhi[(channel),:,:] = MHI
		sample = (mhi, label)
		return sample

	def __len__(self):
		return len(self.FlowLabel)


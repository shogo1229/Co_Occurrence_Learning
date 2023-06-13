from types import LambdaType
import torch, os, glob
import torchvision.transforms as transforms
from PIL import Image
from itertools import chain
import collections
from skimage import io

class MotionDataset(torch.utils.data.Dataset):
	def __init__(self, _imgPaths, _transform=None, _inChannels = 5, _imgHeight=224, _imgWidth=224):
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
		mhi = torch.FloatTensor(self.InChannels*3,self.Height, self.Width)
		for channel in range(self.InChannels):
			if channel == 0:
				cou = 0
			idx = index*self.InChannels+channel
			xImagePath = self.FlowPaths[idx]
			with open(xImagePath, "rb") as f:
				xImage = io.imread(f)
			if self.Transform:
				for c in range(3):
					npimg = xImage[:,:,c].copy()
					img = Image.fromarray(npimg)
					img = self.Transform(img)
					mhi[(cou),:,:] = img
					cou+=1

			#print(xImage.size())
			#print((mhi[(channel),:,:,:]).size())
			
		sample = (mhi, label)
		return sample

	def __len__(self):
		return len(self.FlowLabel)
#torch.Size([3, 224, 224])
#torch.Size([224, 224])
# mhi[(channel),:,:] = xImage
# RuntimeError: expand(torch.FloatTensor{[3, 224, 224]}, size=[224, 224]): the number of sizes provided (2) must be greater or equal to the number of dimensions in the tensor (3)
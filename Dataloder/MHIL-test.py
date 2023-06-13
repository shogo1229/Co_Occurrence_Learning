import os, glob
import pprint as pp
from itertools import chain

trainPath = r"E:\Research\test" # 学習用

print("trainPath:", trainPath)

labelPaths = glob.glob('{}/*'.format(trainPath))
print("LablePaths",labelPaths)
FolderPaths = []
for labelPath in labelPaths:
	FolderPaths.append(glob.glob('{}/*'.format(labelPath)))
imagePaths = []
for FolderPath in FolderPaths:
    for Folder in FolderPath:
        print(Folder)
        imagePaths.append(glob.glob('{}/*.jpg'.format(Folder)))
for imagePath in imagePaths:
    print("imagePath length:",len(imagePath))

labelPaths = list(chain.from_iterable(labelPaths))
print(len(imagePaths))

labelName = [os.path.basename(fn) for fn in glob.glob('{}/*'.format(trainPath))]
print(labelName)

labelNames = [os.path.basename(os.path.dirname(fn)) for fn in labelPaths]
print(len(labelNames), labelNames)
labelIndexs = [labelName.index(l) for l in labelNames]
print(len(labelIndexs),labelIndexs)
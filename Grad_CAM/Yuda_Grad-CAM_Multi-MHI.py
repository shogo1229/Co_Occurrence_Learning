import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import glob
from matplotlib import cm
import re
import torch
from torchvision import models
from PIL import Image
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image

modelPath = r""  # モデルのパス
dataset = r""  # データセットのパス
modelName = r""  # 出力ファイル名

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

class Transform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224, 224]),  # 画像のリサイズ
            transforms.ToTensor()  # テンソルに変換
        ])
    
    def __call__(self, img):
        return self.base_transform(img)

class MobileNet():
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)  # MobileNetV2モデルの読み込み
        self.model.features[0][0] = nn.Conv2d(30, 32, 3)  # 入力チャンネル数を変更
        self.transformClassifier()
        self.transformGrad(True)
    
    def transformClassifier(self):
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1280, 3)  # 分類器の出力次元を変更
        )
    
    def transformGrad(self, boo):
        for p in self.model.features.parameters():
            p.requires_grad = boo

class Cam():
    def __init__(self, loadmodel, target_layer):
        self.model = loadmodel
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.cam_extractor = GradCAMpp(self.model.eval(), target_layer)
    
    def GradCam(self, image):
        output = self.model((image.unsqueeze(0)).to(self.device))
        pred = F.softmax((self.model((image.unsqueeze(dim=0)).to(self.device))))
        pred_idx = pred.max(1)[1]
        cam = self.cam_extractor(output.squeeze(0).argmax().item(), output)
        result = to_pil_image(cam[0], mode="F")
        pred_value = pred.data[0].tolist()
        return result, pred_idx, pred_value
    
    def __call__(self, image):
        result_img, pred_idx, pred_value = self.GradCam(image)

Mobilenet = MobileNet()
model = Mobilenet.model.cuda()
checkpoint = torch.load(modelPath)
model.load_state_dict(checkpoint['model_state_dict'])
grad = GradCAMpp(model.eval(), model.features[18][2])  # Grad-CAMを適用するターゲットレイヤーの指定
transform = Transform()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

list = glob.glob(dataset + "/*")
list.sort(key=natural_keys)
for idx, image in enumerate(list):
    if idx == 0:
        first_Image = Image.open(image)
        # first_Image = first_Image.convert('L')  # グレースケールに変換
        Temporal_tensor = transform(first_Image)
        input_tensor = Temporal_tensor
        print(input_tensor.size())
    else:
        loadImage = Image.open(image)
        # loadImage = loadImage.convert('L')  # グレースケールに変換
        Temporal_tensor = transform(loadImage)
        input_tensor = torch.cat([Temporal_tensor, input_tensor], 0)
        print(input_tensor.size())

output = model(input_tensor.unsqueeze(dim=0).to(device))
mask = grad(output.squeeze(0).argmax().item(), output)  # ヒートマップの取得
first_Image = first_Image.convert('RGB')
first_Image = first_Image.resize((224, 224))
overlaymask = to_pil_image(mask[0], mode="F")
overlay = overlaymask.resize((224, 224), resample=Image.BICUBIC)
cmap = cm.get_cmap("jet")
overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
overlay = Image.fromarray(overlay)
# overlay.save("heatmap.jpg")  # ヒートマップの保存
result = overlay_mask(first_Image, to_pil_image(mask[0], mode="F"), alpha=0.5)  # ヒートマップと元画像のオーバーレイ
result.save(str(modelName) + ".jpg")  # 結果の保存

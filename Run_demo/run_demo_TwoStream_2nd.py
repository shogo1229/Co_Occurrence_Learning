
import os
import sys
import glob
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
sys.path.append('../')
from fps import DispFps
from Network.Spatial.MobileNet import MobileNet_V2_Spatial
from Network.Temporal.MobileNet import MobileNet_V2_Temporal
from PIL import Image
from itertools import count

class BaseTransform():
    def __init__(self):
        self.base_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.base_transform(img)


class Run_TwoStream():
    def __init__(self, Threshold=10, Pixel_Max_Value=255, Pixel_Min_Value=0, Tau=5):
        self.classList = ['Boar','Bear','Others']
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = BaseTransform()
        self.THRESHOLD = Threshold
        self.TAU = Tau
        self.PIXEL_MAX_VALUE = Pixel_Max_Value
        self.PIXEL_MIN_VALUE = Pixel_Min_Value
        self.kernel = np.ones((5, 5), np.uint8)
        self.count = 0
        self.first_counter = 0
        self.InferencceMHI_Tensor = 0
        self.MHIlist = []
        self.Reset = []
        self.Reset = (torch.tensor(self.Reset)).to(self.device)

    def cv2pil(self, image_cv):
        image_pil = Image.fromarray(image_cv)
        return image_pil

    def createMHI(self, MHI, frameDifferenceImage, motionHistoryImage):
        # 更新された部分のインデックスを取得(返り値はタプル)
        idx_PIXEL_MAX_VALUE = np.where(
            frameDifferenceImage == self.PIXEL_MAX_VALUE)
        MHI[idx_PIXEL_MAX_VALUE] = self.TAU  # 更新された部分の値にTAU(残すフレーム数)を代入
        idx_TAU = np.where(MHI > self.PIXEL_MIN_VALUE)  # 画素値が1以上の部分のインデックスを取得
        MHI[idx_TAU] = MHI[idx_TAU] - 1  # 画素値が1以上の全て画素値に対して-1
        # 画素値が-1になった部分を全て0にする
        MHI[MHI < self.PIXEL_MIN_VALUE] = self.PIXEL_MIN_VALUE
        motionHistoryImage = MHI/self.TAU*self.PIXEL_MAX_VALUE  # 画素値が1ずつしか違わないと分かりづらいから変換
        motionHistoryImage = np.fix(motionHistoryImage)  # 一個上でfloat型になったから整数に
        motionHistoryImage = motionHistoryImage.astype(
            np.uint8)  # floatで返すと0,255の値で返されるからunit8型に変換
        return motionHistoryImage, MHI

    def run_demo(self, Spatial_model_path, Temporal_model_path, movie_path=0):
        MobileNet_Temporal = MobileNet_V2_Temporal()
        Temporal_model = MobileNet_Temporal.model.cuda()
        Temporal_model.load_state_dict(torch.load(Temporal_model_path))
        Temporal_model.eval()

        MobileNet_Spatial = MobileNet_V2_Spatial()
        Spatial_model = MobileNet_Spatial.model.cuda()
        Spatial_model.load_state_dict(torch.load(Spatial_model_path))
        Spatial_model.eval()

        capture = cv2.VideoCapture(movie_path)
        ret, frameImage = capture.read()  # フレーム読み込み
        MHI = np.zeros((frameImage.shape[0], frameImage.shape[1]), np.uint8)
        motionHistoryImage = np.zeros(
            (frameImage.shape[0], frameImage.shape[1], 1), np.uint8)
        # frameImageをグレースケール化し,t-2の画像とする
        prevImage = cv2.cvtColor(frameImage, cv2.COLOR_BGR2GRAY)
        ret, frameImage = capture.read()  # フレーム読み込み
        # frameImageをグレースケール化し,t-1の画像とする
        prevImage2 = cv2.cvtColor(frameImage, cv2.COLOR_BGR2GRAY)
        ret, frameImage = capture.read()  # フレーム読み込み
        # frameImageをグレースケール化し,tの画像とする
        currentImage = cv2.cvtColor(frameImage, cv2.COLOR_BGR2GRAY)
        dispFps = DispFps()
        while(capture.isOpened()):
            ret, frameImage = capture.read()  # フレーム読み込み
            # frameImageをグレースケール化し,t+1の画像とする
            nextImage = cv2.cvtColor(frameImage, cv2.COLOR_BGR2GRAY)
            ret, frameImage = capture.read()  # フレーム読み込み
            # frameImageをグレースケール化し,t+2の画像とする
            nextImage2 = cv2.cvtColor(frameImage, cv2.COLOR_BGR2GRAY)
            differenceImage1 = cv2.absdiff(
                prevImage2, prevImage)  # (t-1)-(t-2)の差分
            differenceImage2 = cv2.absdiff(
                prevImage, currentImage)  # (t-1)-tの差分
            differenceImage3 = cv2.absdiff(
                currentImage, nextImage)  # t-(t+1)の差分} - tの差分
            differenceImage4 = cv2.absdiff(
                nextImage, nextImage2)  # (t+1)-(t+2)の差分
            ret, differenceImage1 = cv2.threshold(
                differenceImage1, self.THRESHOLD, self.PIXEL_MAX_VALUE, cv2.THRESH_BINARY)
            ret, differenceImage2 = cv2.threshold(
                differenceImage2, self.THRESHOLD, self.PIXEL_MAX_VALUE, cv2.THRESH_BINARY)
            ret, differenceImage3 = cv2.threshold(
                differenceImage3, self.THRESHOLD, self.PIXEL_MAX_VALUE, cv2.THRESH_BINARY)
            ret, differenceImage4 = cv2.threshold(
                differenceImage4, self.THRESHOLD, self.PIXEL_MAX_VALUE, cv2.THRESH_BINARY)
            frameDifferenceImage1 = cv2.bitwise_and(
                differenceImage2, differenceImage1, mask=differenceImage2)  # 差分画像同士のAnd演算
            frameDifferenceImage2 = cv2.bitwise_and(
                differenceImage4, differenceImage3, mask=differenceImage4)  # 差分画像同士のAnd演算
            frameDifferenceImage = cv2.bitwise_or(
                frameDifferenceImage1, frameDifferenceImage2, mask=frameDifferenceImage1)  # 差分画像同士のAnd演算
            frameDifferenceImage = cv2.medianBlur(
                frameDifferenceImage, 3)  # メディアンフィルタでノイズ除去
            frameDifferenceImage = cv2.dilate(
                frameDifferenceImage, kernel=self.kernel, iterations=1)  # オープニング処理
            frameDifferenceImage = cv2.erode(
                frameDifferenceImage, kernel=self.kernel, iterations=1)  # クロージング処理
            motionHistoryImage, MHI = self.createMHI(
                MHI, frameDifferenceImage, motionHistoryImage)
            prevImage = prevImage2.copy()
            prevImage2 = currentImage.copy()
            currentImage = nextImage.copy()
            nextImage = nextImage2.copy()

            PILframe = self.cv2pil(motionHistoryImage)
            resize_img = self.transform(PILframe)
            inputs = resize_img.unsqueeze_(0)

            frame = self.cv2pil(frameImage)
            frame = self.transform(frame)

            if self.first_counter < self.TAU:
                self.MHIlist.append(inputs)
                self.count += 1
                self.first_counter += 1
            else:
                for num in range(self.count - self.TAU, self.count):
                    if num == self.count - self.TAU:
                        InferencceMHI_Tensor = self.MHIlist[num].to(
                            self.device)
                    else:
                        InferencceMHI_Tensor = torch.cat(
                            ((self.MHIlist[num].to(self.device)), InferencceMHI_Tensor), 1)

                output_Temporal = Temporal_model(InferencceMHI_Tensor)
                Temporal_result = output_Temporal.data[0]
                output_Temporal = F.softmax(output_Temporal, dim=1)

                output_spatial = Spatial_model(
                    frame.unsqueeze(dim=0).to(self.device))
                Spatial_result = output_spatial.data[0]
                output_spatial = F.softmax(output_spatial, dim=1)

                TwoStream_result = Temporal_result + Spatial_result
                TwoStream_result = TwoStream_result/2
                output_twostream = F.softmax(TwoStream_result, dim=0)

                self.InferencceMHI_Tensor = torch.zeros_like(self.Reset)

                pred_idx_twostream = output_twostream.max(0)[1]
                twostream_batch_probs, twostream_batch_indices = output_twostream.sort(
                    dim=0, descending=True)

                pred_idx_Temporal = output_Temporal.max(1)[1]
                Temporal_batch_probs, Temporal_batch_indices = output_Temporal.sort(
                    dim=1, descending=True)

                pred_idx_spatial = output_spatial.max(1)[1]
                Spatial_batch_probs, Spatial_batch_indices = output_spatial.sort(
                    dim=1, descending=True)

                blank = np.zeros((650, 750, 3))
                blank += 255

                cv2.putText(blank, str((f"Temporal : {self.classList[pred_idx_Temporal[0].data.item()]}")), (
                    0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
                for probs, indices in zip(Temporal_batch_probs, Temporal_batch_indices):
                    for k in range(3):
                        cv2.putText(blank, str((f"Top-{k + 1} {probs[k]:.2%} {self.classList[indices[k]]}")), (0, 100+(
                            50*k)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)

                cv2.putText(blank, str((f"Spatial : {self.classList[pred_idx_spatial[0].data.item()]}")), (
                    0, 250), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
                for probs, indices in zip(Spatial_batch_probs, Spatial_batch_indices):
                    for k in range(3):
                        cv2.putText(blank, str((f"Top-{k + 1} {probs[k]:.2%} {self.classList[indices[k]]}")), (0, 300+(
                            50*k)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)

                cv2.putText(blank, str((f"Two Stream : {self.classList[pred_idx_twostream.data.item()]}")), (
                    0, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)
                for k in range(3):
                    cv2.putText(blank, str((f"Top-{k + 1} {twostream_batch_probs[k]:.2%} {self.classList[twostream_batch_indices[k]]}")), (0, 500+(
                        50*k)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3, cv2.LINE_AA)
                dispFps.disp(blank)

                cv2.imshow("Result", blank)
                cv2.imshow("MotionHistoryImage", cv2.resize(
                    motionHistoryImage, (600, 600)))
                cv2.imshow("frameImage", frameImage)

                self.count += 1
                self.MHIlist.append(inputs)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    capture.release()
                    cv2.destroyAllWindows()
                    break

    def __call__(self, Spatial_model_path, Temporal_model_path, movie_path):
        self.run_demo(Spatial_model_path=Spatial_model_path,
                      Temporal_model_path=Temporal_model_path, movie_path=movie_path)


test = Run_TwoStream()
test(Spatial_model_path=r"E:\Research\TwoStreamCNN_2nd-Season\Models\G_Model\max_Val_acc_MoileNet_RGB_20BN-6Class_RGB_full_part1.pth",
     Temporal_model_path=r"E:\Research\TwoStreamCNN_2nd-Season\Models\G_Model\max_Val_acc_MobileNet_MHI_20BN-6class_MHI_log5_part1.pth",
     movie_path=0)

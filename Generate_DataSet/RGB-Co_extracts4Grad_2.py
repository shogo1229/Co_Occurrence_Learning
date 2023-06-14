import random
import glob
import shutil
import os
import re
from tqdm import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

class extractImage():
    def __init__(self, RGBpath, CoPath):
        self.classList = (os.listdir(RGBpath))  # RGBパス内のディレクトリのリストを取得
        self.extracts = 500  # 画像の抽出数
        self.RGB_ImageList = [[], [], []]  # RGB画像のリスト
        self.Co_ImageList = [[], [], []]  # Co画像のリスト
        self.num_List = []  # ランダムな数値のリスト

    def random_num(self, lower_limit, upper_limit, extracts_Number):
        while len(self.num_List) < extracts_Number:
            random_num = random.randint(lower_limit, upper_limit)
            if (not random_num in self.num_List) and (random_num < upper_limit):
                self.num_List.append(random_num)
        return self.num_List

    def extracts_Image(self, rgb, Co, RGB_savePath, Co_savePath, idx):
        print(self.classList[idx])  # クラスの名前を表示
        os.mkdir(RGB_savePath + '/' + self.classList[idx])  # RGB保存先のディレクトリを作成
        os.mkdir(Co_savePath + '/' + self.classList[idx])  # Co保存先のディレクトリを作成
        rgb.sort(key=natural_keys)  # 自然な順序でソート
        Co.sort(key=natural_keys)  # 自然な順序でソート
        extractsList = self.random_num(0, (len(rgb) - 1000), self.extracts)  # 画像の抽出リストを生成
        for Image_id in tqdm(range(self.extracts)):  # 進捗バー付きでループ処理
            shutil.copy(str(rgb[extractsList[Image_id]]), RGB_savePath + '/' + self.classList[idx])  # RGB画像をコピー
            shutil.copy(str(Co[extractsList[Image_id]]), Co_savePath + '/' + self.classList[idx])  # Co画像をコピー

    def __call__(self, RGB_Images, Co_Images, RGB_SavePath, Co_SavePath):
        RGB_classList = glob.glob(RGB_Images + "\*")  # RGB画像のパスリストを取得
        Co_classList = glob.glob(Co_Images + "\*")  # Co画像のパスリストを取得
        for idx, (rgb, Co) in enumerate(zip(RGB_classList, Co_classList)):
            rgb_folderList = glob.glob(rgb + "\*")  # RGB画像のディレクトリリストを取得
            Co_folderList = glob.glob(Co + "\*")  # Co画像のディレクトリリストを取得
            self.extracts_Image(rgb_folderList, Co_folderList, RGB_SavePath, Co_SavePath, idx)  # 画像の抽出とコピーを実行

if __name__ == '__main__':
    test = extractImage(RGBpath=r"E:\Research\DataSet\20BN-Jester-6Class\6class_RGB\test",
                        CoPath=r"E:\Research\DataSet\20BN-Jester-6Class\6class_Synthetic\test")

    test(RGB_Images=r"E:\Research\DataSet\20BN-Jester-6Class\6class_RGB\test",
         Co_Images=r"E:\Research\DataSet\20BN-Jester-6Class\6class_Synthetic\test",
         RGB_SavePath=r"E:\Research\DataSet\20BN-Jester-6Class_Grad\GradTest500_RGB",
         Co_SavePath=r"E:\Research\DataSet\20BN-Jester-6Class_Grad\GradTest500_Syntetic")

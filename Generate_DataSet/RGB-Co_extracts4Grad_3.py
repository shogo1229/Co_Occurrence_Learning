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
    def __init__(self, RGBpath, CoPath, CCoPath):
        self.classList = (os.listdir(RGBpath))  # RGBパス内のディレクトリのリストを取得
        self.extracts = 2000
        self.RGB_ImageList = [[], [], [], [], [], []]
        self.Co_ImageList = [[], [], [], [], [], []]
        self.CCo_ImageList = [[], [], [], [], [], []]
        self.num_List = []

    def random_num(self, lower_limit, upper_limit, extracts_Number):
        while len(self.num_List) < extracts_Number:
            random_num = random.randint(lower_limit, upper_limit)
            if (not random_num in self.num_List) and (random_num < upper_limit):
                self.num_List.append(random_num)
        return self.num_List

    def extracts_Image(self, rgb, Co, CCo, RGB_savePath, Co_savePath, CCo_savePath, idx):
        print(self.classList[idx])
        os.mkdir(RGB_savePath + '/' + self.classList[idx])  # RGB_savePath以下にクラス名のディレクトリを作成
        os.mkdir(Co_savePath + '/' + self.classList[idx])  # Co_savePath以下にクラス名のディレクトリを作成
        os.mkdir(CCo_savePath + '/' + self.classList[idx])  # CCo_savePath以下にクラス名のディレクトリを作成
        rgb.sort(key=natural_keys)
        Co.sort(key=natural_keys)
        CCo.sort(key=natural_keys)
        extractsList = self.random_num(0, (len(rgb) - 1000), self.extracts)
        for Image_id in tqdm(range(self.extracts)):
            shutil.copy(str(rgb[extractsList[Image_id]]), RGB_savePath + '/' + self.classList[idx])  # ファイルをコピー
            shutil.copy(str(Co[extractsList[Image_id]]), Co_savePath + '/' + self.classList[idx])  # ファイルをコピー
            shutil.copy(str(CCo[extractsList[Image_id]]), CCo_savePath + '/' + self.classList[idx])  # ファイルをコピー

    def __call__(self, RGB_Images, Co_Images, CCo_Images, RGB_SavePath, Co_SavePath, CCo_SavePath):
        RGB_classList = glob.glob(RGB_Images + "\*")  # RGB_Images以下のディレクトリのリストを取得
        Co_classList = glob.glob(Co_Images + "\*")  # Co_Images以下のディレクトリのリストを取得
        CCo_classList = glob.glob(CCo_Images + "\*")  # CCo_Images以下のディレクトリのリストを取得
        for idx, (rgb, Co, CCo) in enumerate(zip(RGB_classList, Co_classList, CCo_classList)):
            rgb_folderList = glob.glob(rgb + "\*")  # rgb以下のディレクトリのリストを取得
            Co_folderList = glob.glob(Co + "\*")  # Co以下のディレクトリのリストを取得
            CCo_folderList = glob.glob(CCo + "\*")  # CCo以下のディレクトリのリストを取得
            for id, (rgb_folder, Co_folder, CCo_folder) in enumerate(
                    zip(rgb_folderList, Co_folderList, CCo_folderList)):
                self.RGB_ImageList[idx].extend(glob.glob(rgb_folder + "\*"))  # rgb_folder以下の画像ファイルのリストを取得
                self.Co_ImageList[idx].extend(glob.glob(Co_folder + "\*"))  # Co_folder以下の画像ファイルのリストを取得
                self.CCo_ImageList[idx].extend(glob.glob(CCo_folder + "\*"))  # CCo_folder以下の画像ファイルのリストを取得
            self.extracts_Image(self.RGB_ImageList[idx], self.Co_ImageList[idx], self.CCo_ImageList[idx],
                                RGB_SavePath, Co_SavePath, CCo_SavePath, idx)


if __name__ == '__main__':
    test = extractImage(RGBpath=r"",
                        CoPath=r"",
                        CCoPath=r"")

    test(RGB_Images=r"",
         Co_Images=r"",
         CCo_Images=r"",
         RGB_SavePath=r"",
         Co_SavePath=r"",
         CCo_SavePath=r"")

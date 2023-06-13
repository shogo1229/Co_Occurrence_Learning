import glob
from PIL import Image
import os
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# GIFアニメーションを作成
def create_gif(in_dir, out_filename):
    path_list = sorted(glob.glob(os.path.join(*[in_dir, '*']))) # ファイルパスをソートしてリストする
    path_list.sort(key=natural_keys)
    imgs = []                                                   # 画像をappendするための空配列を定義

    # ファイルのフルパスからファイル名と拡張子を抽出
    for i in range(len(path_list)):
        img = Image.open(path_list[i])                          # 画像ファイルを1つずつ開く
        imgs.append(img)                                        # 画像をappendで配列に格納していく

    # appendした画像配列をGIFにする。durationで持続時間、loopでループ数を指定可能。
    imgs[0].save(out_filename,
                 save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=0)

# GIFアニメーションを作成する関数を実行する
create_gif(in_dir=r'E:\Research\DataSet\Wild_Life\4th_Season\Image\Normal_MHI\test\Co-Occurrence\Others\CoOccurrence_19',
         out_filename='Others-CoOccurrence_19-Normal.gif')
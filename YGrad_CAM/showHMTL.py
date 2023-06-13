from PIL import Image
import os

# フォルダのパス
folder_paths = ["F:\研究関連\TwoStreamCNN_2nd-Season\YGrad_CAM\RGB_AROB-Journal_max-Acc", "F:\研究関連\TwoStreamCNN_2nd-Season\YGrad_CAM\CoOcc_AROB-Journal_max-Acc", "F:\研究関連\TwoStreamCNN_2nd-Season\YGrad_CAM\C-CoOcc_AROB-Journal_max-Acc"]

# HTMLファイルのパス
html_path = "output.html"

# HTMLファイルを開く
with open(html_path, "w") as f:

    # HTMLのヘッダーを書き込む
    f.write("<html>\n<head>\n<style>\nimg {max-width: 400px;}\n.column {float: left; width: 16.66%; padding: 5px;}\n.row::after {content: ''; clear: both; display: table;}\n</style>\n</head>\n<body>\n")

    # 画像を2列にまとめてHTMLに書き込む
    f.write("<div class='row'>\n")

    # フォルダごとに処理する
    for folder_path in folder_paths:

        # 元画像フォルダのパス
        input_folder_path = os.path.join(folder_path, "GradCam_Result")

        # 出力画像フォルダのパス
        output_folder_path = os.path.join(folder_path, "Original_Image")

        # フォルダ内のファイル名を取得する
        file_names = os.listdir(input_folder_path)

        # 元画像の列を追加する
        for i, file_name in enumerate(file_names):

            # 元画像のパス
            input_file_path = os.path.join(input_folder_path, file_name)

            # 画像を開く
            input_image = Image.open(input_file_path)

            # 画像をHTMLに書き込む
            f.write(f"<div class='column'><img src='{input_file_path}'></div>\n")

        # 出力画像の列を追加する
        for i, file_name in enumerate(file_names):

            # 出力画像のパス
            output_file_path = os.path.join(output_folder_path, file_name)

            # 画像を開く
            output_image = Image.open(output_file_path)

            # 画像をHTMLに書き込む
            f.write(f"<div class='column'><img src='{output_file_path}'></div>\n")

        # フォルダごとの列の間に改行を挿入する
        if folder_path != folder_paths[-1]:
            f.write("<div style='clear:both'></div>\n")

    f.write("</div>\n")

    # HTMLのフッターを書き込む
    f.write("</body>\n</html>")

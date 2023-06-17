# README

# Co_Occurrence_Learning

作成者：shogo1229  
概要：エッジデバイス上でのリアルタイム動画認識を目指し、空間的情報と動き情報の共起を学習する。  
最終編集日：2023.06.1  
論文：https://www.jstage.jst.go.jp/article/prociiae/2022/0/2022_11/_article/-char/ja/

- [Calc_Accudracy](about:blank#clacaccuracy)
    - Calc_Accuracy_Kfold.py
    - Calc_Accuracy_Spatial.py
    - Calc_Accuracy_Temporal.py
    - Calc_Accuracy_TwoStream.py
- [Dataloder](about:blank#dataloder)
    - MotionHistory_dataloder.py
    - Pseudo_MotionHIstory_dataloder.py
- [DataSet](about:blank#dataset)
    - Movie
    - RGB_total_0322
- [Excel_Result](about:blank#excelresult)
    - mhiresult.xlsx
    - result.xlsx
    - softmax_result.xlsx
- [Generate_DataSet](about:blank#generatedataset)
    - Choise_TwoStreamData.py
    - Choise.py
    - CompressedImage.py
    - dowanload_Dataset.py
    - make_stackImage_CalcAccuracy.py
    - trainval_split.py
    - extracts_Image.py
    - inversion_Movie.py
    - make_Temporal.py
    - make_TwoStream.py
    - make_stackImage.py
    - move_files.py
    - Rename.py
- [Grad_CAM](about:blank#gradcam)
    - resultimg
    - Grad-CAM.py
- [Models](about:blank#models)
    - max_Val_acc_ResNet50_RGB_total_0322_MHI_Old
    - max_Val_acc_VGG16_RGB_total_0322_ALL
- [Motion_History_Image](about:blank#motionhistoryimage)
    - Motion_History_image.py
- [Network](about:blank#network)
    - pycache
    - **Spatial**
        - ResNet.py
        - VGG16.py
    - **Temporal**
        - ResNet.py
- [Run_demo](about:blank#rundemo)
    - run_demo_Spatial.py
    - run_demo_Temporal.py
    - run_demo_TwoStream.py
- [Transforms](about:blank#transfroms)
    - pycache
    - data_preprocessing.py
    - transfomrs.py
- train_Spatial.py
- train_Temporal.py
- Yuda_train_Spatial.py
- Yuda_train_Temporal.py

# Clac_Accuracy
- Calc_Accuracy_kfold.py  
交差検証を行う際に生成されたモデル全てを一度に評価するコード  
- Calc_Accuracy_Spatial.py  
Spatial CNNの精度検証用コード,コード内に検証用データセット・学習済みモデルのパスを記述して動作.  
共起学習の精度検証もこのコードで動作可能．  
混合行列とある程度の精度評価指標をcmd上に表示，混同行列のヒートマップを画像で出力．  
- Calc_Accuracy_Temporal.  
Temporal CNNの精度検証用コード,コード内に検証用データセット・学習済みモデルのパスを記述して動作.  
混合行列とある程度の精度評価指標をcmd上に表示，混同行列のヒートマップを画像で出力．  
- Calc_Accuracy_TwoStream.py  
Two Stream CNNの精度検証用コード，コード内にSpatial,
各モデルの混合行列とある程度の精度評価指標をcmd上に表示，混同行列のヒートマップを画像で出力．  

# Dataloder
- MotionHistory_dataloder.py  
引数にスタック数を指定する．  
Temporal CNNで学習するためのデータローダー，MHIを時系列方向にスタックしてロードする．   
データセットの形式はMHI生成元の動画ごとにフォルダを作成し，その中にMHIを入れる．  
- Pseudo_MotionHIstory_dataloder.py  
引数にスタック数を指定する．  
Temporal CNNで疑似カラーを付与したMHIを用いて学習するためのデータローダー，MHIを時系列方向にスタックしてロードする．  
データセットの形式はMHI生成元の動画ごとにフォルダを作成し，その中にMHIを入れる．  
# Generate_DataSet
- extracts_Image.py  
データセットから指定された数の画像をランダムに抜き出すコード，適当な画像でテストしたいときに使用．  
引数に抽出対象のフォルダパスと，抽出後の画像を保存するフォルダのパスを渡す．
- inversion_Movie.py  
動画を反転させるコード
引数:変換対象の動画が入ったパス,反転した動画の保存先フォルダのパス
- make_stackImage_CalcAccuracy.py  
精度検証しやすい形式で共起画像を生成する．
- make_stackImage.py   
共起画像を生成する．引数にRGB画像が入ったフォルダとMHIが入ったフォルダパス，共起画像保存先のフォルダパスを渡す．  
RGB画像とMHIは1対1で対応させておく，基本的にはmake_TwoStream.pyで生成した構造をそのまま転用可能．  
cv2.addWeightedの値を変えれば透過率(alpha)を変更できる
- make_TwoStream.py  
MHI,RGB画像を生成するコード.MHIはMotion_History_Image.pyかPseudo_Motion_History_Image.pyを使う  
引数:変換対象の動画が入ったフォルダのパス,RGB・MHI・連続してないRGB画像を保存したい場所
- make_TwoStream_fromImage.py  
MHI,RGB画像を生成するコード.MHIはMotion_History_Image.pyかPseudo_Motion_History_Image.pyを使う  
引数:変換対象の画像が入ったフォルダのパス,RGB・MHI・連続してないRGB画像を保存したい場所  
名前の通り，動画から生成するのではなく，画像から生成する．
- Resize.py  
画像の入ったフォルダを渡し，その中身の画像を全て指定した解像度でリサイズする．
- RGB-Co_extracts4Grad_2.  
データセット内からRGB画像と共起画像を対応付けて規定した枚数をランダムに抜き出すコード
- RGB-Co_extracts4Grad_3.py  
データセット内からRGB画像と共起画像，色付き共起画像をそれぞれ対応付けて規定した枚数をランダムに抜き出すコード
- trainval_split.py
指定フォルダ内の画像や動画を指定した割合で分割保存するコード
# Grad_CAM
- Grad-CAM_Multi-MHI.py  
連番画像を用いて学習したTemporal modelの認識結果をGrad-CAMで解析するためのコード，引数に連番画像が入ったパスを渡す．
- Grad-CAM_Spatial.py  
Spatial CNN,共起学習のモデルの認識結果をGrad-CAMで解析するためのコード，引数にデータセット，モデルのパスを渡す．  
結果はexcelに記入され同じディレクトリに生成される．template.xlsxをコピーして内部に記入される．  
記入内容は元画像，CAMの解析結果，各クラスのスコア
- Grad-CAM_Temporal.py  
MHIを単体で学習したモデルの認識結果をGrad-CAMで解析するためのコード，引数にデータセット，モデルのパスを渡す．  
結果はexcelに記入され同じディレクトリに生成される．template.xlsxをコピーして内部に記入される．  
記入内容は元画像，CAMの解析結果，各クラスのスコア
# Motion_history_image
- Motion_History_Image_prev5.py  
MHI生成用コード，差分を取る際に現在から過去に5枚比較する．  
- Motion_History_Image_Pseudo_color_read_image  
疑似カラーを付与したMHI生成コード，動画からではなく画像を元に生成する．  
- Motion_History_Image_Pseudo_color.py  
疑似カラーを付与したMHI生成用コード，動画から生成  
- Motion_History_Image_raad_image.py  
MHI生成用コード，動画からではなく，画像を元に生成する．  
- Motion_History_Image.py  
MHI生成用コード，動画から生成  

# Network

- Spatial  
空間情報に基づく特徴量を抽出する.要するに普通のRGB画像用の分類ネットワーク
    - ResNet.py  
    - VGG16.py  
    - GhostNet.py  
    - MobileNet.py  
    コード本体を弄って分類クラス数を入れる
- Temporal  
動きの情報に基づく特徴量を抽出する.時間方向に連続した画像(MHI)を入力し分類する
    - ResNet.py  
    - MobileNet.py  
    - VGG16.py  
    入力の数値を書き換えて連番で何枚CNNに入力するかを決める

# Run_demo
- fps.py  
fpsを計算し表示させるためのクラス
- run_demo_CoOccurrence.py  
リアルタイムで共起画像認識を行うコード
- run_demo_Spatial.py  
リアルタイムで静止画像認識を行うコード
- run_demo_Temporal.py  
リアルタイムでTemporalCNNによる認識を行うコード
- run_demo_TwoStream.py  
リアルタイムでTwoStreamCNNによる認識を行うコード
# Transfroms
- data_data_preprocessing.py  
transforms.py用の便利クラスが色々書いてある
- transforms.py  
transformで便利なクラス,関数が大量に書いてある

# 学習用コード群
train_[　].py [　]の中に書いてあるのが学習対象，全て学習時のLossとAccをEpochごとに記したグラフが出力される．
- train_Spatial_KFold.py  
交差検証を行うためのコード，対象はSpatial CNN  
- train_Spatial_Resume.py  
何らかのトラブルで学習が途中で止まった時に追加で学習するためのコード，引数には追加学習したいモデルのパスと保存先を指定する．
- train_Spatial.py  
Spatial CNNの学習用コード  
- train_Temporal_1ch_Kfold.py  
交差検証を行うためのコード，対象は通常のMHIを用いたTemporal CNN  
- train_Temporal_1ch_Resume.py  
何らかのトラブルで学習が途中で止まった時に追加で学習するためのコード，引数には追加学習したいモデルのパスと保存先を指定する．  
対象は通常のMHIを用いて学習したTemporal CNNモデル
- train_Temporal.py  
交差検証無しでTemporal CNNの学習を行う．
- train_Temporal_3ch_Kfold.py  
交差検証を行うためのコード，対象は疑似カラーを付与したMHIを用いたTemporal CNN  
- train_Temporal_3ch_Resume.py  
何らかのトラブルで学習が途中で止まった時に追加で学習するためのコード，引数には追加学習したいモデルのパスと保存先を指定する．  
対象は疑似カラーを付与したMHIを用いて学習したTemporal CNNモデル
- train_Temporal_3ch.py  
交差検証無しでTemporal CNNの学習を行う．
# README

# Two Stream Convolutional Neural Network

作成者：yoshida
概要：りどみを色々備忘録がてら書く予定、ある程度進んだらNotionにも上げる。
最終編集日：2023.06.1

# To Do

- ~~RGBとMHIを重ねる用のコードを書く~~ 3.17実装
- ~~データセット生成コードを書き換える~~ 3.31実装
- データローダーを変える
- ~~推論結果をxlsxで出力する奴を書く~~ 4.25実装(xlsxではなくmatplotlibで出力)
- ~~推論用データセットを作成するコードを改修する~~ 5.15実装
- ~~精度検証用コードをSyntheticに対応させる~~ 5.15実装 # 中身
- [Calc_Accudracy](about:blank#clacaccuracy)
    - Calc_Accuracy_Spatial.py
    - Calc_Accuracy_Temporal.py
    - Calc_Accuracy_TwoStream.py
    - Make_Accuracy_Data.py
    - Make_Accuracy_Data_new.py
- [Dataloder](about:blank#dataloder)
    - pycache
    - Optical_dataloder.py
    - Temporal_dataloder.py
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

- Calc_Accuracy_Spatial.py
Spatial CNNの精度検証用コード,コード内に検証用データセット・学習済みモデル・クラスを記述して動作.
混合行列とある程度の精度評価指標をcmd上に表示
- Calc_Accuracy_Temporal.py
Temporal CNNの精度検証用コード,コード内に検証用データセット・学習済みモデル・クラスを記述して動作.
混合行列とある程度の精度評価指標をcmd上に表示
- Calc_Accuracy_TwoStream.py
Spatial・Temporal CNNの最終層のsoftmax関数の値を統合して精度評価を行う
コード内にSpatial Temporal の学習済みモデルのパス,検証用データセットを記述
検証用データセットは連番のMHIが入ったフォルダ名と,RGB画像のファイル名を1対1で対応させる
例：Temporal/**1**/images… , Spatial/**1.jpg**
Spatial Temporal TwoStream の混合行列を表示,TwoStremの統合手法は加算,平均値も実装済み(未使用)
- Make_Accuracy_Data.py
Motion History ImageとRGB画像を動画から生成するコード
生成したMHIとRGB画像は上のCalc_TwoStreamに対応した形で生成される
コード内にMHI・RGBの保存位置,MHI生成パラメータ,生成元の動画のパスを記述**~~※MHIの変換はopencvを使った旧式の変更コード,新しいやつに変更予定~~**　5.16変更した
- Make_Accuracy_Data_new.py
Make_Accuracy_Data.pyのMHI生成を自作したMHI生成コードに変更した奴,他の動作は変えてない # Dataloder
- Optical_dataloder.py
Two Stream CNNの本来の形であるOpticalFlowで学習するためのデータローダー
x方向のOpticalFlowとy方向のOpticalFlowを与える
- Temporal_dataloder.py
Motion Hisotry Imageで学習するためのデータローダー
呼び方：MotionDataset(学習用データセットのパス, transformer, 連番で何枚読み込むか)
学習用データセットは[v_ClassName_g00_c0]の命名規則に従ってフォルダを作成し,中身にMHIの画像を連番で入れる.
クラス別にフォルダを作成せずに,同じフォルダ内にすべてのクラスのフォルダを入れる**※今後改修予定**

# Dataset

### HMDB-51

51クラスの動画を含んだ行動認識用データセット,train,val,testは公式通りに分解
それぞれRGB,MHI,Sytheticに分けて保存 - #### HMDB-51_split1 - #### HMDB-51_split1 - #### HMDB-51_split1 ### KTH 6クラスの動画からなる行動認識データセット - #### train - #### val - #### Accuracy ### NAIST #### NAIST_4Class - ##### Org_Dataset
背景,熊,猪,鹿の4クラスのデータセット,org_Imageから指定数抜き出した - ##### Org_Image 背景,熊,猪,鹿の4クラスの画像を集めたやつ - ##### Quality5_Dataset org_Datasetを読み込んで保存品質5のjpegで書きだしたデータセット #### NAIST_4Class_GAN - ##### Org_Dataset 背景,熊,猪,鹿の4クラスのデータセット,org_Imageから指定数抜き出した - ##### Org_saves 背景,熊,猪,鹿の4クラスの画像を集めたやつ - ##### Quality5_Dataset org_Datasetと同じファイル名で保存品質を弄ったやつを入れたデータセット
pix2pix学習用に同じファイル名で保存 ### RGB_total_0322 #### Images - #### MHI MHIを学習する用の形に成形したデータセットとSynthetic用のMHIの2種類,中身は同じ保存形式が違うだけ
Logは15frame,前後5frame差分 - #### RGB 動画を1frameごとに区切って保存しただけ,保存形式はいつもの形で学習できる奴
- #### Synthetic 上の二つを重ねることで生成したSynthetic画像,logは15でRGB画像に過去15frame分の動きがついてる
#### Movie 動画を画像に変換するにあたってtrainとvalに分けてある

# Movie

- Fiexed_RGB_0321
MHIが生成できるようにトレイルカメラ(固定カメラ)で撮影された害獣の動画一式
3月21日に動画を収集した
- Fiexed_RGB_0321_inversion
上の動画をすべて反転した動画一式
- Fiexed_RGB_total_0321
上記二つの動画集を合わせた動画一式 # Excel_Result
- mhiresult.xlsx
Spatial,Temporal,TwoStream CNNが出力した各クラスの信頼度を一覧で表示
間違えた画像を探すために推論対象のファイルパスも記述してある**※ただしこれは中間発表に間に合わすために人力で作成したもの,今後自動でcsv生成するコードを作成する予定**
- result.xlsx
Grad-CAM.pyで実行した結果を記述
Grad-CAMをかける前の画像,ヒートマップを重ねた画像,各クラスの信頼度,推論結果を表示
画像を表示するためには推論対象の画像を一旦224×224に成形した保存したやつに対して推論する必要がある**※勝手にリサイズするように今後改修予定**
- softmax_result.xlsx
mhiresult.xlsxが初期コードだとsoftmaxかけてない疑惑があったからかけたやつ

# Generate_DataSet

- inversion_Movie.py
動画を反転させるコード
引数:変換対象の動画が入ったパス,反転した動画の保存先フォルダのパス
- make_Temporal.py
Motion_History_Image.pyのコードからインスタンス生成してMHIを生成するコード
- make_TwoStream.py
MHI,RGB画像を生成するコード.MHIはMotion_History_Image.pyを使う
引数:変換対象の動画が入ったフォルダのパス,RGB・MHI・連続してないRGB画像を保存したい場所
- make_StackImage.py
make_TwoStream.pyで生成したMHI,RGB画像を重ねて動き情報と空間情報が共起した画像を生成するコード
引数はRGB,MHIのフォルダパス,共起画像の保存先パス
cv2.addWeightedの値を変えれば透過率(alpha)を変更できる
- Rename.py
make_TwoStreamで生成したMHIのフォルダ名をデータローダーで読み込めるようにファイル名を変更するコード 引数：変換対象のフォルダ,フォルダ名変更フラグ,ファイル名変更フラグ
- trainval_split.py
trainとvalに分割するコード
train,valの保存先のフォルダ
- CompressedImage.py
jpegの保存品質を指定して書き出すコード 引数；元画像パス,劣化保存先パス,劣化前画像保存パス
- Coise_TwoStreamData.py
TwoStreamCNNでの精度評価,CalcAccuracy用のデータセット選択コード
引数にRGB,MHIの保存元フォルダパスとそれぞれの保存先パス
- choise.py
入力されたフォルダの画像を全部読みこんで指定されたフレームごとに保存するコード
- download_Dataset.py
適当なデータセットをダウンロードするコード
- make_AccuracyData_new,py
精度検証用画像を旧式のMHI生成手法じゃなくて自前のMHI生成コードで行うやつ
引数に動画のパス,RGB,MHIの保存パス
- make_stackImage_CalcAccuracy.py
RGB,MHIそれぞれのCalcAccuracy用のデータセットを使って共起画像を生成するコード
- move_files.py
フォルダやファイルを一つ上のディレクトリに移動させるコード,MHI学習用データセットに変換する
- Rename.py
生成したMHIをMHI学習用コードで使えるような形に名前を変更するコード

# Grad_CAM

- resultimg
Grad-CAMの出力結果(ヒートマップ)を入れるフォルダ
- Grad-CAM.py
Grad-CAM本体,xlsxファイルをあらかじめ用意してないと結果が書き込まれない # Models
- max_Val_acc_VGG16_RGB_total_0322_ALL
Syntheticの画像をVGG16で学習したモデル
lrは0.00001,最適化手法はSGD,momentum=0.9
謎の高精度を叩き出した謎モデル,今度Grad-CAMで見てみる予定
- max_Val_acc_ResNet50_RGB_total_0322_MHI_Old
max_Val_acc_VGG16_RGB_total_0322_ALLに使ったMHIをResnet50で学習したモデル
lrは0.00001,最適化手法はSGD,momentum=0.9
Syntheticとの精度比較用に学習した,MHIは旧式じゃなくて自作した奴を使って生成してる # Motion_History_Image
- Motion_History_Image.py
MHIを出力する本体,MHIのパラメータ系はコード本体の_init_を書き換える
createMHIは愚直に全探索する,createMHI_ver2はnumpy使っていい感じに高速化した奴
引数([MoviePath]変換対象のパス,[SaveFlag]MHIを保存するか否か,[DisplayFlag]出力結果を詳しく表示するか否か,
[MHI_SavePath]MHIの保存先パス,[RGB_SavePath]RGB画像の保存先パス)

# Network

- Spatial
空間情報に基づく特徴量を抽出する.要するに普通のRGB画像用の分類ネットワーク
    - ResNet.py
    - VGG16.py
    コード本体を弄って分類クラス数を入れる
- Temporal
動きの情報に基づく特徴量を抽出する.時間方向に連続した画像(MHI)を入力し分類する
    - ResNet.py
    入力の数値を書き換えて連番で何枚CNNに入力するかを決める

# Run_demo

**初期に書いたスパゲッティコード達,いつか気が向けば書き換える**
- ### run_demo_Spatial.py
動画に対してマイフレームSpatial CNNで学習したモデルを使って推論,結果を表示するコード - ### run_demo_Temporal.py
動画に対してマイフレームTemporal CNNで学習したモデルを使って推論,結果を表示するコード - ### run_demo_TwoStream.py
動画に対してマイフレームTwoStream CNNで学習したモデルを使って推論,結果を表示するコード
動作が怪しい

# Transfroms

- data_data_preprocessing.py
transforms.py用の便利クラスが色々書いてある
- transforms.py
transformで便利なクラス,関数が大量に書いてある

# train_Spatial.py

Spatial CNN学習用コード

# train_Temporal.py

Temporal CNN学習用コード 学習用データセットは全クラスを同じディレクトリに置く
フォルダ名：v_クラス名_gn_cn (nは任意の数値)
画像ファイル名:frame00000n (nは任意の数値,6桁の0埋め)

# Yuda_train_Spatial.py

train_Spatial.pyの機能追加版,学習時にlossとaccのグラフを書き出す

# Yuda_train_Temporal.py

train_Temporal.pyの機能追加版,学習時にlossとaccのグラフを書き出す
import cv2 
import numpy as np
import glob

class MotionHistoryImage():
    def __init__(self,Threshold = 10,Pixel_Max_Value=255,Pixel_Min_Value=0,Tau=5):         #初期化
        self.THRESHOLD = Threshold
        self.TAU = Tau
        self.PIXEL_MAX_VALUE = Pixel_Max_Value
        self.PIXEL_MIN_VALUE = Pixel_Min_Value
        self.kernel = np.ones((5,5),np.uint8)
        self.count = 0

    def createMHI(self,MHI,frameDifferenceImage,motionHistoryImage):
        height, width = frameDifferenceImage.shape
        for y in range (height):
            for x in range(width):
                if frameDifferenceImage[y,x] == self.PIXEL_MAX_VALUE:
                    MHI[y,x] = self.TAU
                else:
                    if MHI[y,x] > 0 :
                        MHI[y,x] = MHI[y,x] - 1
                motionHistoryImage[y,x] =  MHI[y,x]/self.TAU*self.PIXEL_MAX_VALUE
        return motionHistoryImage,MHI

    def createMHI_ver2(self,MHI,frameDifferenceImage,motionHistoryImage):
        idx_PIXEL_MAX_VALUE = np.where(frameDifferenceImage == self.PIXEL_MAX_VALUE)                    #更新された部分のインデックスを取得(返り値はタプル)
        MHI[idx_PIXEL_MAX_VALUE] = self.TAU                                                             #更新された部分の値にTAU(残すフレーム数)を代入
        idx_TAU = np.where(MHI > self.PIXEL_MIN_VALUE)                                                  #画素値が1以上の部分のインデックスを取得
        MHI[idx_TAU] = MHI[idx_TAU] - 1                                                                 #画素値が1以上の全て画素値に対して-1
        MHI[MHI<self.PIXEL_MIN_VALUE] = self.PIXEL_MIN_VALUE                                            #画素値が-1になった部分を全て0にする
        motionHistoryImage = MHI/self.TAU*self.PIXEL_MAX_VALUE                                          #画素値が1ずつしか違わないと分かりづらいから変換
        motionHistoryImage = np.fix(motionHistoryImage)                                                 #一個上でfloat型になったから整数に
        motionHistoryImage = motionHistoryImage.astype(np.uint8)                                        #floatで返すと0,255の値で返されるからunit8型に変換
        return motionHistoryImage,MHI

    def __call__(self,ImageList = None ,SaveFlag = False , DisplayFlag = False ,MHI_SavePath = None ,RGB_SavePath = None,CoOcc_SavePath = None):
        for i,image in enumerate(ImageList):
            if i == 0:
                frameImage = cv2.imread(image)
                MHI = np.zeros((frameImage.shape[0], frameImage.shape[1]), np.uint8)
                motionHistoryImage = np.zeros((frameImage.shape[0], frameImage.shape[1], 1), np.uint8)
                prevImage = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                         #frameImageをグレースケール化し,t-2の画像とする
                continue
            elif i == 1:
                frameImage = cv2.imread(image)                                                                #フレーム読み込み
                prevImage2 = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                        #frameImageをグレースケール化し,t-1の画像とする
                continue
            elif i == 2:
                frameImage = cv2.imread(image)                                                                 #フレーム読み込み
                currentImage = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                      #frameImageをグレースケール化し,tの画像とする
                continue
            elif i == 3:
                frameImage = cv2.imread(image)                                                             #フレーム読み込み
                nextImage = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                 #frameImageをグレースケール化し,t+1の画像とする
                continue
            self.count += 1                                                             
            frameImage = cv2.imread(image)                                                         #フレーム読み込み
            if i == len(ImageList):    
                break                                                         
            nextImage2 = cv2.cvtColor(frameImage,cv2.COLOR_BGR2GRAY)                                #frameImageをグレースケール化し,t+2の画像とする
            differenceImage1 = cv2.absdiff(prevImage2,prevImage)                                    #(t-1)-(t-2)の差分
            differenceImage2 = cv2.absdiff(prevImage,currentImage)                                  #(t-1)-tの差分
            differenceImage3 = cv2.absdiff(currentImage,nextImage)                                  #t-(t+1)の差分} - tの差分
            differenceImage4 = cv2.absdiff(nextImage,nextImage2)                                    #(t+1)-(t+2)の差分
            ret,differenceImage1 = cv2.threshold(differenceImage1,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            ret,differenceImage2 = cv2.threshold(differenceImage2,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            ret,differenceImage3 = cv2.threshold(differenceImage3,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            ret,differenceImage4 = cv2.threshold(differenceImage4,self.THRESHOLD,self.PIXEL_MAX_VALUE,cv2.THRESH_BINARY)
            frameDifferenceImage1 = cv2.bitwise_and(differenceImage2,differenceImage1,mask=differenceImage2)                  #差分画像同士のAnd演算
            frameDifferenceImage2 = cv2.bitwise_and(differenceImage4,differenceImage3,mask=differenceImage4)                  #差分画像同士のAnd演算
            frameDifferenceImage = cv2.bitwise_or(frameDifferenceImage1,frameDifferenceImage2,mask=frameDifferenceImage1)    #差分画像同士のAnd演算
            frameDifferenceImage = cv2.medianBlur(frameDifferenceImage,3)                                                     #メディアンフィルタでノイズ除去
            frameDifferenceImage = cv2.dilate(frameDifferenceImage,kernel=self.kernel,iterations=1)                           #オープニング処理
            frameDifferenceImage = cv2.erode(frameDifferenceImage,kernel=self.kernel,iterations=1)                            #クロージング処理
            motionHistoryImage,MHI = self.createMHI_ver2(MHI,frameDifferenceImage,motionHistoryImage)
            motionHistoryImage = cv2.cvtColor(motionHistoryImage, cv2.COLOR_GRAY2BGR)
            #print(motionHistoryImage.shape,frameImage.shape)
            CoOccurrence_Image = cv2.addWeighted(src1=frameImage,alpha=1,src2=motionHistoryImage,beta=0.5,gamma=0)
            if DisplayFlag == True:
                cv2.imshow("differenceImage1",cv2.resize(differenceImage1,(300,300)))
                cv2.imshow("differenceImage2",cv2.resize(differenceImage2,(300,300)))
                cv2.imshow("differenceImage3",cv2.resize(differenceImage3,(300,300)))
                cv2.imshow("differenceImage4",cv2.resize(differenceImage4,(300,300)))
                cv2.imshow("frameDifferenceImage1",cv2.resize(frameDifferenceImage1,(300,300)))
                cv2.imshow("frameDifferenceImage2",cv2.resize(frameDifferenceImage2,(300,300)))
                cv2.imshow("frameDifferenceImage",cv2.resize(frameDifferenceImage,(300,300)))
                cv2.imshow("Motion History Image",motionHistoryImage)
                cv2.imshow("Input Image",cv2.resize(frameImage,(300,300)))
            if SaveFlag == True: 
                cv2.imwrite(str(MHI_SavePath) +'/MHI_'+ str(self.count) + ".jpg",motionHistoryImage )
                cv2.imwrite(str(CoOcc_SavePath) +'/C_CoOcc_'+ str(self.count) + ".jpg",CoOccurrence_Image )
                #cv2.imwrite(str(RGB_SavePath) +'/RGB_'+ str(self.count) + ".jpg",frameImage )
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            prevImage = prevImage2.copy()
            prevImage2 = currentImage.copy()
            currentImage = nextImage.copy()
            nextImage = nextImage2.copy()

##list = glob.glob(r"E:\Research\DataSet\20BN-Jester-V1\test\Doing other things\84"+ "\*")
#print(list)
#test = MotionHistoryImage()
#test(ImageList=list,SaveFlag=True,MHI_SavePath=r"E:\Research\TwoStreamCNN_2nd-Season\Motion_history_image\test")
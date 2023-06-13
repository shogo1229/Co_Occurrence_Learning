from ast import Pass
import random
import glob
import pprint as pp
import shutil
import os
import re
from tqdm import tqdm

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

class extractImage():
    def __init__(self):
        self.classList = ["Bear","Boar","Others"]
        self.extracts = 5000
        self.RGB5_ImageList = [[],[],[]]
        self.MHI5_List =[[],[],[]]
        self.RMHI5_List =[[],[],[]]
        self.HMHI5_List =[[],[],[]]
        self.Co5_ImageList = [[],[],[]]
        self.RCo5_ImageList = [[],[],[]]
        self.HCo5_ImageList = [[],[],[]]
        
        self.MHI10_List =[[],[],[]]
        self.RMHI10_List =[[],[],[]]
        self.HMHI10_List =[[],[],[]]
        self.Co10_ImageList = [[],[],[]]
        self.RCo10_ImageList = [[],[],[]]
        self.HCo10_ImageList = [[],[],[]]
        self.num_List = []
        print(self.classList)

    def random_num(self,lower_limit,upper_limit,extracts_Number):
        while len(self.num_List) < extracts_Number:
            random_num = random.randint(lower_limit,upper_limit)
            if (not random_num in self.num_List) and (random_num < upper_limit):
                self.num_List.append(random_num)
        return self.num_List
    def extracts_Image(self,RGB5,MHI5,RMHI5,HMHI5,Co5,RCo5,HCo5,
                    MHI10,RMHI10,HMHI10,Co10,RCo10,HCo10,
                    RGBS5,MHIS5,RMHIS5,HMHIS5,CoS5,RCoS5,HCoS5,
                    MHIS10,RMHIS10,HMHIS10,CoS10,RCoS10,HCoS10,
                    idx):
        print(self.classList[idx])
        os.mkdir(RGBS5+'/'+self.classList[idx])
        os.mkdir(MHIS5+'/'+self.classList[idx])
        os.mkdir(RMHIS5+'/'+self.classList[idx])
        os.mkdir(HMHIS5+'/'+self.classList[idx])
        os.mkdir(CoS5+'/'+self.classList[idx])
        os.mkdir(RCoS5+'/'+self.classList[idx])
        os.mkdir(HCoS5+'/'+self.classList[idx])

        os.mkdir(MHIS10+'/'+self.classList[idx])
        os.mkdir(RMHIS10+'/'+self.classList[idx])
        os.mkdir(HMHIS10+'/'+self.classList[idx])
        os.mkdir(CoS10+'/'+self.classList[idx])
        os.mkdir(RCoS10+'/'+self.classList[idx])
        os.mkdir(HCoS10+'/'+self.classList[idx])

        RGB5.sort(key=natural_keys)
        MHI5.sort(key=natural_keys)
        RMHI5.sort(key=natural_keys)
        HMHI5.sort(key=natural_keys)
        Co5.sort(key=natural_keys)
        RCo5.sort(key=natural_keys)
        HCo5.sort(key=natural_keys)

        MHI10.sort(key=natural_keys)
        RMHI10.sort(key=natural_keys)
        HMHI10.sort(key=natural_keys)
        Co10.sort(key=natural_keys)
        RCo10.sort(key=natural_keys)
        HCo10.sort(key=natural_keys)

        extractsList = self.random_num(0,(len(RGB5)-1000),self.extracts)
        for Image_id in tqdm(range(self.extracts)):
            #shutil.copy(str(RGB5[extractsList[Image_id]]),RGBS5+'/'+self.classList[idx])
            #shutil.copy(str(MHI5[extractsList[Image_id]]),MHIS5+'/'+self.classList[idx])
            shutil.copy(str(RMHI5[extractsList[Image_id]]),RMHIS5+'/'+self.classList[idx])
            shutil.copy(str(HMHI5[extractsList[Image_id]]),HMHIS5+'/'+self.classList[idx])
            #shutil.copy(str(Co5[extractsList[Image_id]]),CoS5+'/'+self.classList[idx])
            shutil.copy(str(RCo5[extractsList[Image_id]]),RCoS5+'/'+self.classList[idx])
            shutil.copy(str(HCo5[extractsList[Image_id]]),HCoS5+'/'+self.classList[idx])

            #shutil.copy(str(MHI10[extractsList[Image_id]]),MHIS10+'/'+self.classList[idx])
            shutil.copy(str(RMHI10[extractsList[Image_id]]),RMHIS10+'/'+self.classList[idx])
            shutil.copy(str(HMHI10[extractsList[Image_id]]),HMHIS10+'/'+self.classList[idx])
            #shutil.copy(str(Co10[extractsList[Image_id]]),CoS10+'/'+self.classList[idx])
            shutil.copy(str(RCo10[extractsList[Image_id]]),RCoS10+'/'+self.classList[idx])
            shutil.copy(str(HCo10[extractsList[Image_id]]),HCoS10+'/'+self.classList[idx])

    def __call__(self,RGB5_Images,MHI5_Images,Rainbow5_MHImages,HOT5_MHImages,Co5_Images,Rainbow5_CoImages,HOT5_CoImages,
                MHI10_Images,Rainbow10_MHImages,HOT10_MHImages,Co10_Images,Rainbow10_CoImages,HOT10_CoImages,
                RGB5_SavePath,MHI5_SavePath,Rainbow5_SavePath,HOT5_SavePath,Co5_SavePath,Rainbow5_CoSavePath,HOT5_CoSavePath,
                MHI10_SavePath,Rainbow10_SavePath,HOT10_SavePath,Co10_SavePath,Rainbow10_CoSavePath,HOT10_CoSavePath):
        print("call in")
        RGB5_classList = glob.glob(RGB5_Images + "\*")
        MHI5_classList = glob.glob(MHI5_Images + "\*")
        RainbowMHI5_classList = glob.glob(Rainbow5_MHImages + "\*")
        HOTMHI5_classList = glob.glob(HOT5_MHImages + "\*")
        Co5_classList = glob.glob(Co5_Images + "\*")
        RainbowCo5_classList =  glob.glob(Rainbow5_CoImages + "\*")
        HOTCo5_classList =  glob.glob(HOT5_CoImages + "\*")

        MHI10_classList = glob.glob(MHI10_Images + "\*")
        RainbowMHI10_classList = glob.glob(Rainbow10_MHImages + "\*")
        HOTMHI10_classList = glob.glob(HOT10_MHImages + "\*")
        Co10_classList = glob.glob(Co10_Images + "\*")
        RainbowCo10_classList =  glob.glob(Rainbow10_CoImages + "\*")
        HOTCo10_classList =  glob.glob(HOT10_CoImages + "\*")

        for idx,(rgb5,mhi5,rmhi5,hmhi5,co5,rco5,hco5,mhi10,rmhi10,hmhi10,co10,rco10,hco10) in enumerate(zip(RGB5_classList,MHI5_classList,RainbowMHI5_classList,HOTMHI5_classList,Co5_classList,RainbowCo5_classList,HOTCo5_classList,MHI10_classList,RainbowMHI10_classList,HOTMHI10_classList,Co10_classList,RainbowCo10_classList,HOTCo10_classList)):
            rgb5_folderList = glob.glob(rgb5 + "\*")
            mhi5_folderList = glob.glob(mhi5 + "\*")
            rmhi5_folderList = glob.glob(rmhi5 + "\*")
            hmhi5_folderList = glob.glob(hmhi5 + "\*")
            co5_folderList = glob.glob(co5 + "\*")
            rco5_folderList = glob.glob(rco5 + "\*")
            hco5_folderList = glob.glob(hco5 + "\*")

            mhi10_folderList = glob.glob(mhi10 + "\*")
            rmhi10_folderList = glob.glob(rmhi10 + "\*")
            hmhi10_folderList = glob.glob(hmhi10 + "\*")
            co10_folderList = glob.glob(co10 + "\*")
            rco10_folderList = glob.glob(rco10 + "\*")
            hco10_folderList = glob.glob(hco10 + "\*")

            for id,(rgb5_folder,mhi5_folder,rmhi5_folder,hmhi5_folder,co5_folder,rco5_folder,hco5_folder,mhi10_folder,rmhi10_folder,hmhi10_folder,co10_folder,rco10_folder,hco10_folder) in enumerate(zip(rgb5_folderList,mhi5_folderList,rmhi5_folderList,hmhi5_folderList,co5_folderList,rco5_folderList,hco5_folderList,mhi10_folderList,rmhi10_folderList,hmhi10_folderList,co10_folderList,rco10_folderList,hco10_folderList)):
                self.RGB5_ImageList[idx].extend(glob.glob(rgb5_folder + "\*"))
                self.MHI5_List[idx].extend(glob.glob(mhi5_folder + "\*"))
                self.RMHI5_List[idx].extend(glob.glob(rmhi5_folder + "\*"))
                self.HMHI5_List[idx].extend(glob.glob(hmhi5_folder + "\*"))
                self.Co5_ImageList[idx].extend(glob.glob(co5_folder + "\*"))
                self.RCo5_ImageList[idx].extend(glob.glob(rco5_folder + "\*"))
                self.HCo5_ImageList[idx].extend(glob.glob(hco5_folder + "\*"))

                self.MHI10_List[idx].extend(glob.glob(mhi10_folder + "\*"))
                self.RMHI10_List[idx].extend(glob.glob(rmhi10_folder + "\*"))
                self.HMHI10_List[idx].extend(glob.glob(hmhi10_folder + "\*"))
                self.Co10_ImageList[idx].extend(glob.glob(co10_folder + "\*"))
                self.RCo10_ImageList[idx].extend(glob.glob(rco10_folder + "\*"))
                self.HCo10_ImageList[idx].extend(glob.glob(hco10_folder + "\*"))

            self.extracts_Image(self.RGB5_ImageList[idx],self.MHI5_List[idx],self.RMHI5_List[idx],self.HMHI5_List[idx],self.Co5_ImageList[idx],self.RCo5_ImageList[idx],self.HCo5_ImageList[idx],
                                self.MHI10_List[idx],self.RMHI10_List[idx],self.HMHI10_List[idx],self.Co10_ImageList[idx],self.RCo10_ImageList[idx],self.HCo10_ImageList[idx],
                                RGB5_SavePath,MHI5_SavePath,Rainbow5_SavePath,HOT5_SavePath,Co5_SavePath,Rainbow5_CoSavePath,HOT5_CoSavePath,
                                MHI10_SavePath,Rainbow10_SavePath,HOT10_SavePath,Co10_SavePath,Rainbow10_CoSavePath,HOT10_CoSavePath,idx)
if __name__ == '__main__':
    test = extractImage()
    test(RGB5_Images=r"I:\Wild-Life4th\Normal_MHI_log5\test\RGB",
        MHI5_Images = r"I:\Wild-Life4th\Normal_MHI_log5\test\MHI",
        Rainbow5_MHImages =r"I:\Wild-Life4th\Normal_MHI_log5\test\Color_MHI",
        HOT5_MHImages=r"I:\Wild-Life4th\Normal_MHI_log5\test\HOT-MHI",
        Co5_Images=r"I:\Wild-Life4th\Normal_MHI_log5\test\Co-Occurrence",
        Rainbow5_CoImages=r"I:\Wild-Life4th\Normal_MHI_log5\test\Color_Co-Occurrence",
        HOT5_CoImages = r"I:\Wild-Life4th\Normal_MHI_log5\test\HOT-Co-Occurrence",
        MHI10_Images = r"I:\Wild-Life4th\Normal_MHI_log10\test\MHI",
        Rainbow10_MHImages =r"I:\Wild-Life4th\Normal_MHI_log10\test\Color-MHI",
        HOT10_MHImages=r"I:\Wild-Life4th\Normal_MHI_log10\test\HOT-MHI",
        Co10_Images=r"I:\Wild-Life4th\Normal_MHI_log10\test\Co-Occurrence",
        Rainbow10_CoImages=r"I:\Wild-Life4th\Normal_MHI_log10\test\Color-Co-Occurrence",
        HOT10_CoImages = r"I:\Wild-Life4th\Normal_MHI_log10\test\HOT-Co-Occurrence",
        RGB5_SavePath=r"I:\Wild-Life_4th_Grad-CAM\tau5\RGB",
        MHI5_SavePath = r"I:\Wild-Life_4th_Grad-CAM\tau5\MHI",
        Rainbow5_SavePath =r"I:\Wild-Life_4th_Grad-CAM\tau5\MHI-R",
        HOT5_SavePath =r"I:\Wild-Life_4th_Grad-CAM\tau5\MHI-I",
        Co5_SavePath=r"I:\Wild-Life_4th_Grad-CAM\tau5\Co",
        Rainbow5_CoSavePath =r"I:\Wild-Life_4th_Grad-CAM\tau5\Co-R",
        HOT5_CoSavePath =r"I:\Wild-Life_4th_Grad-CAM\tau5\Co-I",
        MHI10_SavePath = r"I:\Wild-Life_4th_Grad-CAM\tau10\MHI",
        Rainbow10_SavePath =r"I:\Wild-Life_4th_Grad-CAM\tau10\MHI-R",
        HOT10_SavePath =r"I:\Wild-Life_4th_Grad-CAM\tau10\MHI-I",
        Co10_SavePath=r"I:\Wild-Life_4th_Grad-CAM\tau10\Co",
        Rainbow10_CoSavePath =r"I:\Wild-Life_4th_Grad-CAM\tau10\Co-R",
        HOT10_CoSavePath =r"I:\Wild-Life_4th_Grad-CAM\tau10\Co-I"
        )

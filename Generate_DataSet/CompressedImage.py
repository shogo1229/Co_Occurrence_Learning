import cv2
import glob
import os
import pprint as pp

class make_CompressedImage():
    def __init__(self):
        self.className = []
    def image_path(self,image_path,save_path,org_save_path):
        for x,class_file in enumerate(image_path):
            print("class Name : ",self.className[x])
            compressed_Image_path = str(save_path) + "//" + str(self.className[x])
            Orginal_Image_path = str(org_save_path) + "//" + str(self.className[x])
            os.mkdir(compressed_Image_path)
            os.mkdir(Orginal_Image_path)
            image_list = glob.glob(class_file + '\*')
            for i,image in enumerate(image_list):
                Compressed_save_image = str(compressed_Image_path + "//" + str(self.className[x]+"_"+str(i)+".jpg"))
                Org_save_image = str(Orginal_Image_path + "//" + str(self.className[x]+"_"+str(i)+".jpg"))
                self.compress_image(org_image=image,save_path=Compressed_save_image,Org_save_path = Org_save_image )
    def compress_image(self,org_image,save_path,Org_save_path):
        img = cv2.imread(org_image)
        img = cv2.resize(img,(224,224))
        cv2.imwrite(save_path,img,[int(cv2.IMWRITE_JPEG_QUALITY),5])
        cv2.imwrite(Org_save_path,img)
    def __call__(self,image_path=None,save_path=None,org_save_path = None):
        self.image_path(glob.glob(image_path+'\*'),save_path,org_save_path)

test = make_CompressedImage()
test(image_path=r"",
save_path=r"",
org_save_path = r"")

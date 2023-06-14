import os
import glob
import re
from tqdm import tqdm
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

dir_path = r""
dir_list = glob.glob(dir_path+"/*")
classL = os.listdir(dir_path)
for x,classlist in enumerate(dir_list):
    mlist = glob.glob(classlist+"/*")
    mlist.sort(key=natural_keys)
    for i,movie in tqdm(enumerate(mlist)):
        #print(str(dir_path) +"/"+ str(classL[x])+"//RGB_Co_"+str(i)+".jpg",movie)
        os.rename(movie,str(dir_path) +"/"+ str(classL[x])+"//RGB_"+str(i)+".jpg")
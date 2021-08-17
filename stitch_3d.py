import cv2,os,glob
import json
import time
import numpy as np
from skimage import io
from sklearn import metrics as mr



def read_image(img_path):
    img = io.imread(img_path)
    img = np.array(img)
    return img

base_dir = r'F:\PMJ\python\data\A1-新鲜样本\拼接3d/shg'
target_dirs = os.listdir(base_dir)
img_list = []
for target_dir in target_dirs:
    src_path = os.path.join(base_dir, target_dir, 'CellVideo/stitch.tif')
    if not os.path.exists(src_path):
        continue
    img = read_image(src_path)
    img_list.append(img) #  [垂直张数，水平张数]'

img_3d = np.zeros((len(img_list),9063,11925), dtype=np.uint16)
for i in range(len(img_list)):
    img_3d[i,:,:] = img_list[i]
io.imsave(base_dir+r'/img_3d.tif', img_3d, check_contrast=False)
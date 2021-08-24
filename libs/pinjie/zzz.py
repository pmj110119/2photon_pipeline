import numpy as np
from skimage import io
import cv2
def read_image(img_path, rotate="fixedly"):
    print(img_path)
    ret, img = cv2.imreadmulti(img_path, [], cv2.IMREAD_ANYCOLOR)
    #img = io.imread(img_path)
    print(type(img))
    # if rotate == "fixedly":
    #     img = np.array(img)
    # elif rotate == "clockwise":
    #     img = np.rot90(np.array(img), k=1, axes=(2, 1))
    # elif rotate == "anticlockwise":
    #     img = np.rot90(np.array(img), k=1, axes=(1, 2))
    return img
img_list = [r'../..\data\CellVideo 0.tif']

#img_list = [r'F:\PMJ\python\2photon_pipeline-master\data\A1-新鲜样本\拼接2D\fresh-changzi-shg 2021-08-10 19-34-32\CellVideo\AVG_CellVideo 0.tif']
read_image(img_list[0])
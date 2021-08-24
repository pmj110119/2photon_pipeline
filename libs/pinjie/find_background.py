import numpy as np
from skimage import io
import time
import matplotlib.pyplot as plt
import stitch_line_search
from sklearn import metrics as mr
from tqdm import tqdm
import cv2

img_in_path1 = r"E:\compose\中日友好\lung\fuction\c19\fad_780_350_0.6\CellVideo\CellVideo stitch mode2.tif"
img_in_path2 = r"E:\compose\中日友好\lung\fuction\c19\nadh_780_350_0.6\CellVideo\CellVideo stitch mode2.tif"
img_out_path = r"E:\compose\中日友好\lung\fuction\c19\div.tif"


def read_image(img_path):
    img = io.imread(img_path)
    img = np.array(img)
    return img


# 前三分之一全部置成一个数
def mean_background(img_path):
    img = read_image(img_path)
    print(img.reshape(-1).shape)
    pixel_list = sorted(img.reshape(-1).tolist())
    l = len(pixel_list)
    # plt.plot(pixel_list[int(l * 0.1):int(l * 1)])
    # plt.ylim(0, 65536)
    # plt.show()
    # 根据荧光稀疏度决定系数
    mean_b = pixel_list[int(l * 0.2)]
    return mean_b


def NADH_div_FAD(img1_path, img2_path):
    mean_b1 = mean_background(img1_path)
    mean_b2 = mean_background(img2_path)

    img1 = read_image(img1_path)
    img2 = read_image(img2_path)

    img1[img1 < mean_b1] = mean_b1 + 1
    img2[img2 < mean_b2] = mean_b2 + 1

    div = img1 / img2 * 50  # (img1 - mean_b1) / (img2 - mean_b2)
    div = np.array(div, dtype=np.uint8)
    io.imsave(img_out_path, div, check_contrast=False)

    return 0


def main():
    NADH_div_FAD(img_in_path1, img_in_path2)


if __name__ == '__main__':
    main()

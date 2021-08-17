import numpy as np
from skimage import io, filters
import matplotlib.pyplot as plt
import numba as nb
import cv2
import find_background


"""
----------
bias1 : 背景偏差大小（0-150），正常取1
bias2 : 允许增强比例小于1，对亮区域进行抑制
bias3 : 减小荧光增强的量，避免误将背景增强，需要权衡
enhance_per : 手动提高增强比例，正常取1
----------
"""

img_in_path = r"E:\compose\中日友好\muscle\b19_muscle\af42\CellVideo\CellVideo.tif"
img_out_path = r"E:\compose\中日友好\muscle\b19_muscle\af42\CellVideo\CellVideo light.tif"


def read_image(img_path):
    img = io.imread(img_path)
    img = np.array(img)
    return img


# bias1:背景值偏差大小; bias2:允许比例小于1，对亮区域进行抑制; bias3减小荧光增强的量，可以避免误将背景增强
bias1 = 1
bias2 = 1
bias3 = 50
enhance_per = 1
BackgroundMean = find_background.mean_background(img_in_path)


def get_Fluorescence_distribution(img):
    for i in range(2):
        img = cv2.GaussianBlur(img, (51, 51), 100)
    for i in range(5):
        img = cv2.GaussianBlur(img, (151, 151), 100)
    # 减去背景
    img = img - BackgroundMean + bias1
    # 获得比例图
    img_f = (np.ones(img.shape, np.float32) * (np.max(img)) - bias2) / img

    return img_f


def compensation(img):
    img_ = merge(img)
    img_f = get_Fluorescence_distribution(img_)
    img_f = img_f * enhance_per
    print("最大增强比例:", np.max(img_f))

    img[img < BackgroundMean + bias1] = BackgroundMean + bias1 + 1
    img_b = img - BackgroundMean - bias1

    # img_c = (img - BackgroundMean - bias1 - bias3) * img_f + BackgroundMean
    img_c = img_b * img_f + BackgroundMean
    print(img_c.shape)
    return np.array(img_c, img.dtype)


def merge(img):
    selected = []
    img_mean = []

    for i in range(img.shape[0]):
        img_mean.append(np.mean(img[i]))
    limit = sorted(img_mean)[len(img_mean) // 2]
    print(limit)

    counter = 0
    while counter < len(img_mean) // 2:
        num = np.random.randint(0, len(img_mean))
        if np.mean(img[num]) > limit:
            selected.append(num)
            counter += 1

    print(len(selected))

    img_counter = 0
    img_ = np.zeros(img.shape[1:3], np.float32)
    for i in range(len(selected)):
        img_ += (img[selected[i]] - img_) / (img_counter + 1)
        img_counter += 1

    return img_


def main():


    img = read_image(img_in_path)

    # img = get_Fluorescence_distribution(img[330])
    # img_f = get_Fluorescence_distribution(img_)

    img_result = compensation(img)

    io.imsave(img_out_path, img_result, check_contrast=False)


if __name__ == '__main__':
    main()

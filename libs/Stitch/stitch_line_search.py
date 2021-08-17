import numpy as np
from skimage import io
import cv2
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

img_in_path1 = r'E:\compose\test111.tif'
img_in_path2 = r'E:\compose\test222.tif'

"""
----------
top_start : 行搜索起点
search_boundary : 搜索边界，与设置的模板长度有关
----------
"""

top_start = 5
search_boundary = 5


def read_image(img_path, rotate="fixedly"):
    img = io.imread(img_path)
    if rotate == "fixedly":
        img = np.array(img)
    elif rotate == "clockwise":
        img = np.rot90(img, k=1, axes=(2, 1))
    elif rotate == "anticlockwise":
        img = np.rot90(img, k=1, axes=(1, 2))
    return img


# 获取梯度图
def preprocessing(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return sobelx, sobely


def search_line(pre_img1x, pre_img2x, pre_img1y, pre_img2y):
    global key_point, correlation12_row
    y, x = pre_img1x.shape[:]
    model = [1, 2, 3, 2, 1]
    line = []
    # 选取起点
    for i in range(x - top_start*2):
        correlation12_row = []
        i = i + top_start - len(model)//2
        dot1x = np.dot(pre_img1x[0:1, i:i + 5], model)
        dot2x = np.dot(pre_img2x[0:1, i:i + 5], model)
        dot1y = np.dot(pre_img1y[0:1, i:i + 5], model)
        dot2y = np.dot(pre_img2y[0:1, i:i + 5], model)

        correlation12 = np.abs((dot1x - dot2x + 0.01) / (dot1x + dot2x + 0.01)) + \
                        np.abs((dot1y - dot2y + 0.01) / (dot1y + dot2y + 0.01))
        correlation12_row.append(correlation12)
    line.append(correlation12_row.index(min(correlation12_row)) + len(model) + top_start)

    # 选取剩余分割线
    for i in range(y):
        i = i + 1
        correlation12_row = []

        for j in range(len(model)):
            j = j + line[i - 1] - len(model)
            dot1x = np.dot(pre_img1x[i:i + 1, j:j + len(model)], model)
            dot2x = np.dot(pre_img2x[i:i + 1, j:j + len(model)], model)
            dot1y = np.dot(pre_img1y[i:i + 1, j:j + len(model)], model)
            dot2y = np.dot(pre_img2y[i:i + 1, j:j + len(model)], model)

            correlation12 = np.abs((dot1x - dot2x + 0.01) / (dot1x + dot2x + 0.01)) + \
                            np.abs((dot1y - dot2y + 0.01) / (dot1y + dot2y + 0.01))
            correlation12_row.append(correlation12)

        key_point = correlation12_row.index(min(correlation12_row)) - len(model)//2 + line[i - 1]

        # keypoint的边缘检测应该根据重叠区域大小
        if key_point > x-search_boundary:
            line.append(key_point - len(model)//2)
        elif key_point < search_boundary:
            line.append(key_point + len(model)//2)
        else:
            line.append(key_point)

    return line


def show_line(img, line):
    y, x = img.shape[0:2]
    for i in range(y):
        img[i][line[i]] = 65520
    return img


def mix_img(img1, img2, line):
    y, x = img1.shape[0:2]
    for i in range(y - 2):
        img1[i:i + 1, line[i]:x] = img2[i:i + 1, line[i]:x]

    # 暂时不平滑缝合线
    # kernel = [[0.095, 0.118, 0.095], [0.118, 0.148, 0.118], [0.095, 0.118, 0.095]]
    # for j in range(y - 3):
    #     j = j + 1
    #     for i in range(3):
    #         img1[j:j + 1, line[j] + i:line[j] + i + 1] = int(
    #             np.sum(np.multiply(img1[j - 1:j + 2, line[j] - 1 + i:line[j] + 2 + i], kernel)))

    return img1


def mix_img_2(img1, img2):
    y, x = img1.shape[0:2]
    for i in range(y - 2):
        img1[i:i + 1, x // 2:x] = img2[i:i + 1, x // 2:x]
    return img1


def main():
    img_out_path = r'E:\compose\test_mix_smooth.tif'

    sobelx1, sobely1 = preprocessing(read_image(img_in_path1))
    sobelx2, sobely2 = preprocessing(read_image(img_in_path2))

    img1 = read_image(img_in_path1)
    img2 = read_image(img_in_path2)

    stitch_line = search_line(sobelx1, sobelx2, sobely1, sobely2)
    # io.imsave(img_out_path1, show_line(img1, stitch_line), check_contrast=False)
    # io.imsave(img_out_path2, show_line(img2, stitch_line), check_contrast=False)
    io.imsave(img_out_path, mix_img(img1, img2, stitch_line), check_contrast=False)

    print("wanle")
    plt.plot(stitch_line)
    plt.show()
    for i in range(len(stitch_line)):
        print(stitch_line[i])


if __name__ == '__main__':
    main()

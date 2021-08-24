import numpy as np
from skimage import io
import time
import matplotlib.pyplot as plt
from sklearn import metrics as mr
from tqdm import tqdm
import cv2

"""
----------
offset_limit : 行搜索起点
overlap_limit : 搜索边界，与设置的模板长度有关
img_avg_num : frame/page
grid_shape : 拍图大小
----------
"""

offset_limit = 5
overlap_limit = 100

img_avg_num = 1
grid_shape = [50, 50]


def read_image(img_path, rotate="fixedly"):

    img = io.imread(img_path)
    if rotate == "fixedly":
        img = np.array(img)
    elif rotate == "clockwise":
        img = np.rot90(img, k=1, axes=(2, 1))
    elif rotate == "anticlockwise":
        img = np.rot90(img, k=1, axes=(1, 2))
    return img


def img_merge(img_array, num_per_pos=1):
    def _img_invalid(img):
        valid_flag = True
        row, col = img.shape[0:2]

        # Case: too many 65535
        count_65535 = np.sum(np.where(img == 65535))
        if count_65535 > row * col * 0.1:
            valid_flag = False

    def _merge(img_stack2merge):
        merged = np.zeros(img_stack2merge.shape[1:], np.float32)
        img_counter = 0
        for i in range(img_stack2merge.shape[0]):
            if _img_invalid(img_stack2merge[i]):
                merged += (img_stack2merge[i] - merged) / (img_counter + 1)
                img_counter += 1
        if not img_counter:
            merged = img_stack2merge[0]
        return np.array(merged, img_array.dtype)

    num_img = img_array.shape[0] // num_per_pos
    out_img_shape = (num_img,) + img_array.shape[1:]
    out_img_dtype = img_array.dtype
    img_merged = np.zeros(out_img_shape, out_img_dtype)

    for j in range(num_img):
        img_merged[j] = _merge(img_array[j * num_per_pos: (j + 1) * num_per_pos])
    return img_merged


def computer_overlap(img1, img2, offset):
    mutual_info_score = []
    img_y, img_x = img1.shape[0:2]
    # 上移重叠互信息
    if offset < 0:
        for overlap in range(overlap_limit):
            overlap = overlap + 1
            overlap_first = np.reshape(img1[0:img_y + offset, img_x - overlap:img_x], -1)
            overlap_next = np.reshape(img2[-offset:img_y, 0:overlap], -1)
            mutual_info = mr.mutual_info_score(overlap_first, overlap_next)
            print(mutual_info)
            mutual_info_score.append(mutual_info)
    else:
        # 下移重叠互信息
        for overlap in range(overlap_limit):
            overlap = overlap + 1
            overlap_first = np.reshape(img1[offset:img_y, img_x - overlap:img_x], -1)
            overlap_next = np.reshape(img2[0:img_y - offset, 0:overlap], -1)
            mutual_info = mr.mutual_info_score(overlap_first, overlap_next)
            # print(mutual_info)
            mutual_info_score.append(mutual_info)
    plt.plot(mutual_info_score)
    plt.show()
    global key_point
    for i in range(len(mutual_info_score)-3):
        if (mutual_info_score[i + 1] - mutual_info_score[i]) / (
                mutual_info_score[i + 2] - mutual_info_score[i + 1]) < 0:
            key_point = max(mutual_info_score[i:overlap_limit-3])
            break

    overlap_x = (mutual_info_score.index(key_point) + 1) if key_point in mutual_info_score else 11
    # print("---------", overlap_x)

    return overlap_x


def histequ(img, nlevels=65536):
    histogram = np.bincount(img.flatten(), minlength=nlevels)

    uniform_hist = (nlevels - 1) * (np.cumsum(histogram) / (img.size * 1.0))
    uniform_hist = uniform_hist.astype('uint16')

    height, width = img.shape
    uniform_gray = np.zeros(img.shape, dtype='uint16')
    for i in range(height):
        for j in range(width):
            uniform_gray[i, j] = uniform_hist[img[i, j]]
    return uniform_gray


def img_enhance(img):
    image = histequ(img)
    image = (image / 256).astype(np.uint8)
    image = cv2.bilateralFilter(src=image, d=3, sigmaColor=500, sigmaSpace=500)
    image = cv2.bilateralFilter(src=image, d=3, sigmaColor=500, sigmaSpace=500)
    return image


def outliner_handing(overlap_x, offset_y):
    overlap_x = (np.array(overlap_x)).reshape(grid_shape[0], grid_shape[1] - 1)
    offset_y = (np.array(offset_y)).reshape(grid_shape[0], grid_shape[1] - 1)
    print(overlap_x)
    overlap_x_avg = []
    offset_y_avg = []

    for i in range(grid_shape[0]-1):
        # 暂时不考虑奇数行和偶数行的区别，统一处理试试
        # 如果均值-最小值<5，说明没有异常值,出现异常值，重叠和偏移都会异常
        if np.mean(overlap_x[:, i]) - min(overlap_x[:, i]) < 5 and max(overlap_x[:, i]) - np.mean(overlap_x[:, i]) < 5:
            overlap_x_avg.append(int(np.sum(overlap_x[:, i]) // grid_shape[0]))
            offset_y_avg.append(int(np.sum(offset_y[:, i]) // grid_shape[0]))
        else:
            overlap_x_avg.append(int(np.median(overlap_x[:, i])))
            offset_y_avg.append(int(np.median(offset_y[:, i])))

    return overlap_x_avg, offset_y_avg


def main():
    img_in_path = r'E:\compose\muscle\af-3\CellVideo\CellVideoNoExposure0merged.tif'
    offset_row = 0
    img_0_filp = False
    img = read_image(img_in_path, rotate="fixedly")
    img = np.flip(img, 1) if img_0_filp else img
    all_overlap = []

    # 逐张计算重叠量，剔除异常值
    img_merged = img_merge(img, img_avg_num)
    for row in tqdm(range(grid_shape[0])):
        overlap_x = []
        print("\n", "第{}行偏移量计算开始".format(row + 1))
        # 偶数行，正向
        if row % 2 == 0:
            for i in range(grid_shape[1] - 1):
                img_merged_row = img_merged[row * grid_shape[1]:(row + 1) * grid_shape[1]]

                img_1f = img_enhance(img_merged_row[i])
                img_2f = img_enhance(img_merged_row[i + 1])
                # 正向的时候，左边是图一，右边是图二。反向的话，右边图一，左边图二。
                overlap_x_ = computer_overlap(img_1f, img_2f, offset_row)  # mode3下，行内翻转改这里
                overlap_x.append(overlap_x_)
        # 奇数行，反向
        elif row % 2 == 1:
            print("进入反向计算")
            for i in range(grid_shape[1] - 1):
                i = grid_shape[1] - 2 - i  # 取图反向
                img_merged_row = img_merged[row * grid_shape[1]:(row + 1) * grid_shape[1]]

                img_1f = img_enhance(img_merged_row[i])
                img_2f = img_enhance(img_merged_row[i + 1])
                overlap_x_ = computer_overlap(img_2f, img_1f, offset_row)
                overlap_x.append(overlap_x_)
        print("第{}行偏移量计算完成".format(row + 1))
        all_overlap = all_overlap + overlap_x

        print(overlap_x)
    print(all_overlap)


if __name__ == '__main__':
    main()


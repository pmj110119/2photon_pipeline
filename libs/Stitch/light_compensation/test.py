import numpy as np
from skimage import io, filters
import os


def read_image(img_path):
    img = io.imread(img_path)
    img = np.array(img)
    return img


def split(path):
    img = read_image(path)
    h, x, y = img.shape
    for i in range(h):
        path_out = r"E:\compose\muscle\af-3\CellVideo\1\each_img\img{}.tif".format(i)
        io.imsave(path_out, img[i], check_contrast=False)


def comp(path, path_out):
    path1 = r"E:\compose\muscle\shg-3\CellVideo\CellVideo 0.tif"
    img1 = read_image(path1)
    shape = (5292, 512, 512)
    a = np.zeros(shape, dtype=img1.dtype)
    count = 0
    for i in os.listdir(path):
        img_path = r"E:\compose\muscle\af-3\CellVideo\1\each_img\{}".format(i)
        img = read_image(img_path)
        a[count] = img
        count += 1
    io.imsave(path_out, a, check_contrast=False)
    return 0


def main():
    path1 = r"E:\compose\中日友好\muscle\b19_muscle\shg42\CellVideo\CellVideo 0.tif"
    path2 = r"E:\compose\中日友好\muscle\b19_muscle\shg42\CellVideo\CellVideo 1.tif"
    path_out = r"E:\compose\中日友好\muscle\b19_muscle\shg42\CellVideo\CellVideo.tif"
    img1 = read_image(path1)
    img2 = read_image(path2)
    shape = (5292, 512, 512)
    a = np.zeros(shape, dtype=img1.dtype)
    print(img1.shape)
    a[0:5000] = img1
    a[5000:5292] = img2
    io.imsave(path_out, a, check_contrast=False)

    # path = r"E:\compose\muscle\af-3\CellVideo\1\CellVideo.tif"
    # split(path)

    # path = r"E:\compose\muscle\af-3\CellVideo\1\each_img"
    # path_out = r"E:\compose\muscle\af-3\CellVideo\1\result.tif"
    # comp(path, path_out)


if __name__ == '__main__':
    main()

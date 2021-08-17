import cv2,os,glob
import json
import time
import numpy as np
from skimage import io
import stitch_line_search
from sklearn import metrics as mr

f = open("configuration file.json", "r")
content = f.read()
config = json.loads(content)

grid_shape = config['grid_shape']
img_avg_num = config['img_avg_num']
overlap_limit = config['overlap_limit']

offset_row = config['pre_offset_row']
offset_col = config['pre_offset_col']

# overlap_row_avg1 = config['pre_overlap_row_1'][0:grid_shape[1]]
# overlap_row_avg2 = config['pre_overlap_row_2'][0:grid_shape[1]]
# pre_overlap_col = config['pre_overlap_col'][0:grid_shape[0]]


def read_image(img_path, rotate='fixedly'):
    img = io.imread(img_path)
    if rotate == 'fixedly':
        img = np.array(img)
    elif rotate == 'clockwise':
        img = np.rot90(np.array(img), k=1, axes=(2, 1))
    elif rotate == 'anticlockwise':
        img = np.rot90(np.array(img), k=1, axes=(1, 2))
    return img


def img_merge(img_array, num_per_pos=4):
    def _img_invalid(img):
        valid_flag = True
        row, col = img.shape[0:2]

        # Case: too many 65535
        count20000 = np.sum(img[:, 400:512] > 18000)
        # print(count20000)

        # if np.mean(img) > 7600:
        #     valid_flag = False

        if count20000 > 512 * 10:
            valid_flag = False
            # print(1111)

        return valid_flag

    def _merge(img_stack2merge):
        merged = np.zeros(img_stack2merge.shape[1:], np.float32)
        img_counter = 0
        for i in range(img_stack2merge.shape[0]):
            if _img_invalid(img_stack2merge[i]):
                merged += (img_stack2merge[i] - merged) / (img_counter + 1)
                img_counter += 1
            # else:
            #     io.imshow(img_stack2merge[i])
            #     plt.show()
        if img_counter == 0:
            img_all_error = img_stack2merge[0]
            img_all_error[img_all_error > 0] = 5900
            merged = img_all_error
            print("三帧都不对")
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
    global key_point
    for i in range(len(mutual_info_score)):
        if (mutual_info_score[i + 1] - mutual_info_score[i]) / (
                mutual_info_score[i + 2] - mutual_info_score[i + 1]) < 0:
            key_point = max(mutual_info_score[i:overlap_limit])
            break
    overlap_x = mutual_info_score.index(key_point) + 1

    return overlap_x


# 全部采用从左往右拼接，overlap和offset不区分xy
def stitching(img, overlap, offset, filp=False):
    if filp:
        img = np.flip(img, 0)
    else:
        img = img

    img_y, img_x = img.shape[1:3]
    img_out_shape = (img_y + 1500, img_x * grid_shape[1] + 500)
    img_dtype = img.dtype
    img_out = np.zeros(img_out_shape, img_dtype)

    img_out[0 + 500:img_y + 500, 0 + 500:img_x + 500 - overlap[0] // 2] = img[0][:, 0:img_x - overlap[
        0] // 2]  # img_x - overlap_x[0]//2

    for i in range(grid_shape[1] - 1):
        if i <= grid_shape[0] - 3:
            img_out[0 + offset * (i + 1) + 500: img_y + offset * (i + 1) + 500,
            (1 + i) * img_x - sum(overlap[0:i + 1]) + 500 + overlap[i] // 2: (2 + i) * img_x - sum(
                overlap[0:i + 1]) + 500 - overlap[i + 1] // 2] = \
                img[i + 1][:, overlap[i] // 2:img_x - overlap[i + 1] // 2]
        # 最后一张右侧不需要裁剪
        else:
            img_out[0 + offset * (i + 1) + 500: img_y + offset * (i + 1) + 500,
            (1 + i) * img_x - sum(overlap[0:i + 1]) + 500 + overlap[i] // 2:(2 + i) * img_x - sum(
                overlap[0:i + 1]) + 500] = \
                img[i + 1][:, overlap[i] // 2:img_x]

    # 注意此处是上下都减去了偏移量，是因为存在上下偏移，以后标定好偏移方向后，可以只减一遍
    img_out = img_out[500 + np.abs(offset * (grid_shape[1] - 1)):500 + img_y - np.abs(offset * (grid_shape[1] - 1)),
              500: img_x * grid_shape[1] - np.sum(overlap) + 500]

    return img_out


def histequ(img, nlevels=65536):
    histogram = np.bincount(img.flatten(), minlength=nlevels)

    uniform_hist = (nlevels - 1) * (np.cumsum(histogram) / (img.size * 1.0))
    uniform_hist = uniform_hist.astype('uint16')

    height, width = img.shape
    img_dtype = img.dtype
    uniform_gray = np.zeros(img.shape, img_dtype)
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


def stitching_smooth(img_merged, overlap_avg, flip=False, img_0_filp=False, smooth=True):
    #print(f'Overlap: {overlap_avg}, in line filp: {flip}')
    img_merged = np.flip(img_merged, 0) if flip else img_merged
    img_merged = np.flip(img_merged, 1) if img_0_filp else img_merged

    img_y, img_x = img_merged.shape[1:3]
    img_row = stitching(img_merged, overlap_avg, offset_row, filp=False)

    if smooth:
        for i in range(grid_shape[1] - 1):
            img1 = img_merged[i][-offset_row * i:img_y + offset_row * (grid_shape[0] - 1 - i),
                   img_x - overlap_avg[i]:img_x]
            img2 = img_merged[i + 1][int(offset_row * (-i - 1)):int(img_y + offset_row * (grid_shape[0] - 2 - i)),
                   0:overlap_avg[i]]

            sobelx1, sobely1 = stitch_line_search.preprocessing(img1)
            sobelx2, sobely2 = stitch_line_search.preprocessing(img2)

            stitch_line = stitch_line_search.search_line(sobelx1, sobelx2, sobely1, sobely2)
            mix_img = stitch_line_search.mix_img(img1, img2, stitch_line)
            mix_img = mix_img[-offset_row * (grid_shape[0] - 1):mix_img.shape[0], :]

            if i == 0:
                img_row[:, img_x - overlap_avg[0]:img_x] = mix_img
            else:
                img_row[:, img_x * (i + 1) - np.sum(overlap_avg[0:i + 1]):img_x * (i + 1) - np.sum(overlap_avg[0:i])] = mix_img

    return img_row


def stitching_main(mode, img_in_path, img_out_path, smooth=True, configs = []):
    """
    ----------
    img_in_path：输入地址
    img_out_path：输出地址

    <stitching mode>
    mode1：右下→左上
    mode2：左下→右上
    mode3：左上→右下
    mode4：右上→左下
    smooth：缝合线模式

    <configuration>
    rotate: rotate image singly
    img_0_flip: flip image singly
    col_flip: flip column when stitching
    row1_flip: flip all the even row(stand for first line), to deal with s-capture
    row2_flip: flip all the odd row(stand for second line), to deal with s-capture
    ----------
    """
    global img_avg_num
    global grid_shape
    global overlap_row_avg1
    global overlap_row_avg2

    [img_avg_num, grid_shape, overlap_row_avg1, overlap_row_avg2, pre_overlap_col] = configs
    # Read configuration files
    rotate = config[mode]['rotate']
    row1_flip = config[mode]['row1_flip']
    row2_flip = config[mode]['row2_flip']
    col_flip = config[mode]['col_flip']
    img_0_flip = config[mode]['img_0_flip']

    # Read image and preprocess
    time_start = time.time()
    img = read_image(img_in_path, rotate=rotate)  # clockwise --> anticlockwise
    img_y, img_x = img.shape[1:3]

    overlap_max = max(np.sum(overlap_row_avg1), np.sum(overlap_row_avg2))
    img_dtype = img.dtype
    # img_all_row = np.zeros(
    #     [grid_shape[1], img_x - 2 * np.abs(offset_row * (grid_shape[1] - 1)), img_y * grid_shape[1] - overlap_max],
    #     img_dtype)
    img_all_row = np.zeros(
        [grid_shape[0], img_x - 2 * np.abs(offset_row * (grid_shape[1] - 1)), img_y * grid_shape[1] - overlap_max],
        img_dtype)

  
    img_merged = img_merge(img, img_avg_num)
    for row in range(grid_shape[0]):
        img_merged_row = img_merged[row * grid_shape[1]:(row + 1) * grid_shape[1]]
        if row % 2 == 0:
            img_all_row[row] = stitching_smooth(img_merged_row, overlap_row_avg1, row1_flip, img_0_flip, smooth=smooth)[:,
                               0:img_y * grid_shape[1] - overlap_max]

        elif row % 2 == 1:
            img_all_row[row] = stitching_smooth(img_merged_row, overlap_row_avg2, row2_flip, img_0_flip, smooth=smooth)[:,
                               0:img_y * grid_shape[1] - overlap_max]
        #print(f'Row/Column {row+1} finished')

    # 计算行与行之间的重叠量
    # overlap_y = []
    # for i in tqdm(range(grid_shape[0] - 1)):
    #     img_1f = np.rot90(img_enhance(img_all_row[i]), k=1, axes=(0, 1))
    #     img_2f = np.rot90(img_enhance(img_all_row[i + 1]), k=1, axes=(0, 1))
    #     # 正向的时候，左边是图一，右边是图二。反向的话，右边图一，左边图二。
    #     overlap_y_ = computer_overlap(img_1f, img_2f, offset_col)
    #     overlap_y.append(overlap_y_)
    # print("keyyyyyyyyyyyyyyyyyyyyyyyy", overlap_y)

    # print("计算行之间的重叠量和偏移量，重叠量：{}".format(pre_overlap_col))


    grid_shape[0], grid_shape[1] = 1, grid_shape[0]  # Grid shape changed
    img_out = stitching_smooth(np.rot90(img_all_row, k=1, axes=(1, 2)), pre_overlap_col, flip=col_flip, smooth=smooth)
    img_out = np.rot90(img_out, k=-1)

    io.imsave(img_out_path, img_out, check_contrast=False)
    time_end = time.time()

    print('--- Finished. ', "Time cost:", time_end - time_start, 's')
    

    return img_out


if __name__ == '__main__':
    base_dir = r'F:\PMJ\python\data\A1-新鲜样本\拼接3d'
    target_dirs = os.listdir(base_dir)
    config_list = []
    for target_dir in target_dirs:
        src_path = os.path.join(base_dir, target_dir, 'CellVideo/CellVideo 0.tif')
        save_path = os.path.join(base_dir, target_dir, 'CellVideo/CellVideo 0_拼接结果.tif')
        if not os.path.exists(src_path):
            continue
        config_list.append(['mode3', src_path, save_path, 3, [19, 25]]) #  [垂直张数，水平张数]


    for mode, img_in_path, img_out_path, avg_num, grid_sh in config_list:
        print(img_in_path)
        # Update parameter
        img_avg_num = avg_num 
        grid_shape = grid_sh


        overlap_row_avg1 = [250] * grid_shape[1]
        overlap_row_avg2 = [250] * grid_shape[1]
        pre_overlap_col = np.array([250] * grid_shape[0])

        configs = [img_avg_num, grid_shape, overlap_row_avg1, overlap_row_avg2, pre_overlap_col]
        stitching_main(mode, img_in_path, img_out_path, smooth=True, configs = configs)
        continue
        # Stitch
        try:
            stitching_main(mode, img_in_path, img_out_path, smooth=True, configs = configs)
        except:
            print('ERROR!!!!')

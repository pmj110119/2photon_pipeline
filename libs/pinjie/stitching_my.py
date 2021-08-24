import numpy as np
from skimage import io
import time
import matplotlib.pyplot as plt
import stitch_line_search
# from sklearn import metrics as mr
# from tqdm import tqdm
import cv2,os,shutil
import json
# import numba as nb

f = open(r"configuration file.json", "r")
content = f.read()
config = json.loads(content)
grid_shape = config["grid_shape"]
img_avg_num = config["img_avg_num"]
overlap_limit = config["overlap_limit"]
offset_row = config["pre_offset_row"]
offset_col = config["pre_offset_col"]

overlap_row_avg1 = config["pre_overlap_row_1"][0:grid_shape[1]]
overlap_row_avg2 = config["pre_overlap_row_2"][0:grid_shape[1]]
pre_overlap_col = config["pre_overlap_col"][0:grid_shape[0]]

print(overlap_row_avg1)
# for i in range(35):
#     img_col = grid_shape[0]
#     overlap_avg = overlap_row_avg1
#     print(img_col * (i + 1) - np.sum(overlap_avg[0:i]))
# exit()
def read_image(img_path, rotate="fixedly"):
    print(img_path)

    # shutil.copyfile(img_path,os.path.basename(img_path))

    # ret, imgs = cv2.imreadmulti(os.path.basename(img_path), [], cv2.IMREAD_ANYCOLOR)
    # print(os.path.basename(img_path))
    # print(len(imgs))

    # os.remove(os.path.basename(img_path))  
    # img = np.zeros((len(imgs), 512, 512))
    # for i in range(len(imgs)):
    #     img[i,:,:] = imgs[i]
    img = io.imread(img_path)

    # print(np.shape(img))
    # print(type(img))
    # curve = np.zeros((512,512*100),np.uint16)
    # for i in range(100,200,3):
    #     curve[:,i*512*3:(i*3+1)*512] = img[j]
    # cv2.imwrite('111.png',curve)
    # exit()

    # if rotate == "fixedly":
    #     img = np.array(img)
    # elif rotate == "clockwise":
    #     img = np.rot90(np.array(img), k=1, axes=(2, 1))
    # elif rotate == "anticlockwise":
    #     img = np.rot90(np.array(img), k=1, axes=(1, 2))
    return img


def img_merge(img_array, num_per_pos=4):
    def _img_invalid(img):
        valid_flag = True
        # row, col = img.shape[0:2]

        # Case: too many 65535
        count20000 = np.sum(img[400:512,:] > 40000)

        # if np.mean(img) > 7600:
        #     valid_flag = False

        if count20000 > 512 * 10:
            valid_flag = False

        return valid_flag

    def _merge(img_stack2merge):
        merged = np.zeros(img_stack2merge.shape[1:],np.float)
        img_counter = 0
        for i in range(img_stack2merge.shape[0]):
            # if _img_invalid(img_stack2merge[i]):
            #     merged =merged+img_stack2merge[i]
            #     img_counter =img_counter+ 1
                # merged=merged/img_counter
            if _img_invalid(img_stack2merge[i]):
                merged += (img_stack2merge[i] - merged) / (img_counter + 1)
                img_counter += 1
            # else:
            #     io.imshow(img_stack2merge[i])
            #     plt.show()
        if img_counter == 0:
            img_all_error = img_stack2merge[1]
            # img_all_error[img_all_error > 0] = 5900
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
    img_row, img_col = img1.shape[0:2]
    # 上移重叠互信息
    if offset < 0:
        for overlap in range(overlap_limit):
            overlap = overlap + 1
            overlap_first = np.reshape(img1[0:img_row + offset, img_col - overlap:img_col], -1)
            overlap_next = np.reshape(img2[-offset:img_row, 0:overlap], -1)
            mutual_info = mr.mutual_info_score(overlap_first, overlap_next)
            print(mutual_info)
            mutual_info_score.append(mutual_info)
    else:
        # 下移重叠互信息
        for overlap in range(overlap_limit):
            overlap = overlap + 1
            overlap_first = np.reshape(img1[offset:img_row, img_col - overlap:img_col], -1)
            overlap_next = np.reshape(img2[0:img_row - offset, 0:overlap], -1)
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
    # print('len:',len(img))
   

    img_row, img_col = img.shape[1:3]
    img_out_shape = (img_row + 1500, img_col * grid_shape[1] + 500)
    img_dtype = img.dtype
    img_out = np.zeros(img_out_shape, img_dtype)

    img_out[0 + 500:img_row + 500, 0 + 500:img_col + 500 - overlap[0] // 2] = img[0][:, 0:img_col - overlap[
        0] // 2]  # img_col - overlap_x[0]//2

    for i in range(grid_shape[1] - 1):
        if i <= grid_shape[1] - 3:
            if i<=len(img)-3:
                #print(i, len(img), len(overlap))
                row_base = offset * (i + 1) + 500
                col_low = (1 + i) * img_col - sum(overlap[0:i + 1]) + 500 + overlap[i] // 2
                col_upper = (2 + i) * img_col - sum(overlap[0:i + 1]) + 500 - overlap[i + 1] // 2

                img_out_value = img[i + 1][:, overlap[i] // 2:img_col - overlap[i + 1] // 2]
                img_out[row_base: img_row + row_base,col_low: col_upper ] = img_out_value
                    
        # 最后一张右侧不需要裁剪
        else:
            try:
                img_out[0 + offset * (i + 1) + 500: img_row + offset * (i + 1) + 500,
                (1 + i) * img_col - sum(overlap[0:i + 1]) + 500 + overlap[i] // 2:(2 + i) * img_col - sum(
                    overlap[0:i + 1]) + 500] = \
                    img[i + 1][:, overlap[i] // 2:img_col]
            except:
                pass

    #定义好offset，可以直接裁剪
    img_out = img_out[500 + np.abs(offset * (grid_shape[1] - 1)):500 + img_row - np.abs(offset * (grid_shape[1] - 1)),
              500: img_col * grid_shape[1] - np.sum(overlap) + 500]
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


def stitching_smooth(img_merged_channel1,img_merged_channel2, overlap_avg, filp=False, img_0_filp=False, smooth=True):
    # print(" 重叠量：", overlap_avg, ",行内翻转：", filp)
    img_merged_channel1 = np.flip(img_merged_channel1, 0) if filp else img_merged_channel1
    img_merged_channel1 = np.flip(img_merged_channel1, 1) if img_0_filp else img_merged_channel1
    img_merged_channel2 = np.flip(img_merged_channel2, 0) if filp else img_merged_channel2
    img_merged_channel2 = np.flip(img_merged_channel2, 1) if img_0_filp else img_merged_channel2

    img_row, img_col = img_merged_channel1.shape[1:3]
    row_img_channel1 = stitching(img_merged_channel1, overlap_avg, offset_row, filp=False)
    row_img_channel2 = stitching(img_merged_channel2, overlap_avg, offset_row, filp=False)

    if smooth:
        for i in range(grid_shape[1] - 1):
            img1_channel1 = img_merged_channel1[i][-offset_row * i:img_row + offset_row * (grid_shape[0] - 1 - i),
                   img_col - overlap_avg[i]:img_col]
            img2_channel1 = img_merged_channel1[i + 1][int(offset_row * (-i - 1)):int(img_row + offset_row * (grid_shape[0] - 2 - i)),
                   0:overlap_avg[i]]

            img1_channel2 = img_merged_channel2[i][-offset_row * i:img_row + offset_row * (grid_shape[0] - 1 - i),
                   img_col - overlap_avg[i]:img_col]
            img2_channel2 = img_merged_channel2[i + 1][int(offset_row * (-i - 1)):int(img_row + offset_row * (grid_shape[0] - 2 - i)),
                   0:overlap_avg[i]]

            sobelx1, sobely1 = stitch_line_search.preprocessing(img1_channel1)
            sobelx2, sobely2 = stitch_line_search.preprocessing(img2_channel1)

            stitch_line = stitch_line_search.search_line(sobelx1, sobelx2, sobely1, sobely2)
            mix_img_channel1 = stitch_line_search.mix_img(img1_channel1, img2_channel1, stitch_line)#图像channel1
            mix_img_channel1 = mix_img_channel1[-offset_row * (grid_shape[0] - 1):mix_img_channel1.shape[0], :]

            mix_img_channel2 = stitch_line_search.mix_img(img1_channel2, img2_channel2, stitch_line)#图像channel2
            mix_img_channel2 = mix_img_channel2[-offset_row * (grid_shape[0] - 1):mix_img_channel2.shape[0], :]

            if i == 0:
                row_img_channel1[:, img_col - overlap_avg[0]:img_col] = mix_img_channel1
                row_img_channel2[:, img_col - overlap_avg[0]:img_col] = mix_img_channel2
            else:
                try:
                    #print(i,img_col * (i + 1) - np.sum(overlap_avg[0:i + 1]),img_col * (i + 1) - np.sum(overlap_avg[0:i]), mix_img_channel1.shape)
                    row_img_channel1[:, img_col * (i + 1) - np.sum(overlap_avg[0:i + 1]):img_col * (i + 1) - np.sum(overlap_avg[0:i])] = mix_img_channel1
                    row_img_channel2[:, img_col * (i + 1) - np.sum(overlap_avg[0:i + 1]):img_col * (i + 1) - np.sum(overlap_avg[0:i])] = mix_img_channel2
                except:
                    #print('报错了，但我先不管，诶就是玩儿')
                    pass
    return row_img_channel1,row_img_channel2
def sitch_image(mode,smooth,img_in_path1,img_out_path1,img_in_path2,img_out_path2):
    """
    ----------
    img_in_path：输入地址
    img_out_path：输出地址

    <拼接模式>
    mode1：右下→左上
    mode2：左下→右上
    mode3：左上→右下
    mode4：右上→左下
    smooth：缝合线模式
    ----------
    """
    if mode == "mode1":
        rotate = config["mode1"]["rotate"]
        row1_flip = config["mode1"]["row1_flip"]
        row2_flip = config["mode1"]["row2_flip"]
        img_0_flip = config["mode1"]["img_0_flip"]
    elif mode == "mode2":
        rotate = config["mode2"]["rotate"]
        row1_flip = config["mode2"]["row1_flip"]
        row2_flip = config["mode2"]["row2_flip"]
        img_0_flip = config["mode2"]["img_0_flip"]
    elif mode == "mode3":
        rotate = config["mode3"]["rotate"]
        row1_flip = config["mode3"]["row1_flip"]
        row2_flip = config["mode3"]["row2_flip"]
        img_0_flip = config["mode3"]["img_0_flip"]
    else:
        rotate = config["mode4"]["rotate"]
        row1_flip = config["mode4"]["row1_flip"]
        row2_flip = config["mode4"]["row2_flip"]
        img_0_flip = config["mode4"]["img_0_flip"]

    time_start = time.time()
    img_channel1 = read_image(img_in_path1,rotate=rotate)
    img_channel2 = read_image(img_in_path2,rotate=rotate) 

    img_row, img_col = img_channel1.shape[1:3]
    
   
    for ele in overlap_row_avg1:
        ele=1
    for ele in overlap_row_avg2:
        ele=1
    overlap_max = max(np.sum(overlap_row_avg1), np.sum(overlap_row_avg2))
    img_dtype = img_channel1.dtype
    

    img_all_row_channel1 = np.zeros(
        [grid_shape[0], img_col - 2 * np.abs(offset_row * (grid_shape[1] - 1)), img_row * grid_shape[1] - overlap_max],
        img_dtype)
        
    img_all_row_channel2 = np.zeros(
        [grid_shape[0], img_col - 2 * np.abs(offset_row * (grid_shape[1] - 1)), img_row * grid_shape[1] - overlap_max],
        img_dtype)
    
    img_merged_channel1 = img_merge(img_channel1, img_avg_num)
    img_merged_channel2 = img_merge(img_channel2, img_avg_num)

    curve_channel1 = np.zeros((512*grid_shape[1]),512*grid_shape[0]))
    for row in range(grid_shape[0]):
        img_merged_row_channle1 = img_merged_channel1[row * grid_shape[1]:(row + 1) * grid_shape[1]]#拿出每一行对应的图像
        img_merged_row_channle2 = img_merged_channel2[row * grid_shape[1]:(row + 1) * grid_shape[1]]#拿出每一行对应的图像
        

    print("所有行拼接完成，准备行之间的拼接。")
    

    print("2、行间拼接开始。")
    img_out_channel1,img_out_channel2 = stitching_smooth(np.rot90(img_all_row_channel1,k=1, axes=(1, 2)),np.rot90(img_all_row_channel2,k=1,axes=(1,2)), pre_overlap_col,smooth=smooth)

    io.imsave(img_out_path1, img_out_channel1, check_contrast=False)
    io.imsave(img_out_path2, img_out_channel2, check_contrast=False)
    print("整体拼接完成。")
    time_end = time.time()
    print("time cost:", time_end - time_start, 's')





if __name__ == '__main__':
    # img_list=['b2','b4','b5','b35','b36','b37','c2','c8','c9','c10','c18','c19','c23']
    # img_list=['c13','c27']
    # img_list=['c8']
    # img_list=[r'D:\中日友好\拼接\A-ss-lung-c105-755-1.62-200\CellVideo\CellVideo 0.tif',r'D:\中日友好\拼接\A-ss-lung-c105-755-2.04-200-2\CellVideo\CellVideo 0.tif',
    img_list=[r'F:\PMJ\python\2photon_pipeline-master\data\A1-新鲜样本\拼接3d\fresh-changzi-AF-3D-23.729\CellVideo/CellVideo 0.tif']
    #img_list=[r'../../data/AF\CellVideo 0.tif']

    # img_in_path1=r'D:\中日友好\拼接\A-ss-lung-c105-755-1.62-200\CellVideo\CellVideo 0.tif'
    # img_out_path1=r'D:\中日友好\拼接\A-ss-lung-c105-755-1.62-200\CellVideo\sitch.tif'  
    for i in range(len(img_list)):
        # temp_path_fad=img_in_path1.replace('shg',img_list[i])
        # temp_path_fad_save=img_out_path1.replace('shg',img_list[i])
        # temp_path_nadh=temp_path_fad.replace('shg','af')
        # temp_path_nadh_save=temp_path_fad_save.replace('shg','af')
        temp_path_fad=img_list[i]
        temp_path_fad_save=temp_path_fad.replace('CellVideo 0','sitch')
        temp_path_nadh=temp_path_fad.replace('AF','shg')
        temp_path_nadh_save=temp_path_fad_save.replace('AF','shg')
        sitch_image(mode='mode1',smooth=True,img_in_path1=temp_path_fad,img_in_path2=temp_path_nadh,img_out_path1=temp_path_fad_save,img_out_path2=temp_path_nadh_save)
        print(i)
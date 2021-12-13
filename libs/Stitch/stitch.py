# coding: utf-8
from os import path
import numpy as np
from scipy import ndimage
from skimage import io


def stitch(img_path)
    images = io.imread(img_path)*2#.astype(np.float32)
    print(images.max()/65535.0) # TODO: 将放大系数30改为自适应系数，让灰度级别尽可能大点，但不要越界，即超过65535
    # exit()
    row_num,col_num = 30,31
    rows,cols = [],[]
    for row in range(row_num):
        rows += [row]*col_num
        if row%2==0:
            cols += list(range(col_num))
        else:
            cols += list(range(col_num-1,-1,-1))
    # # exit()
    # rows = [0]*22+[1]*22
    # cols = list(range(22))+list(range(21,-1,-1))

    # cols=[0,1,2,0,1,2]
    # rols=[0,0,0,1,1,1]
    print(images.shape,images.dtype)

    zzz = images#[[2,3,4,43-2,43-3,43-4],:,:]
    output = np.zeros((zzz.shape[1]*row_num,zzz.shape[2]*col_num),dtype=images.dtype)

    h,w = images.shape[1], images.shape[2]
    hide = 51
    step = zzz.shape[1] - hide

    overlay_map = np.zeros_like(output,dtype=np.uint8)
    for i in range(zzz.shape[0]):
        row,col = rows[i], cols[i]
        x0,x1 = step*col, step*col+w
        y0,y1 = step*row, step*row+h

        image = zzz[i,:,:]
        patch = output[y0:y1,x0:x1]

        #print('[%d,%d]'%(x0,x1),output.shape,patch.shape)

        overlay_patch = overlay_map[y0:y1,x0:x1]
    

        if overlay_patch.sum() != 0:
            #print(patch.max(),image.max())

            #patch = transfer(patch,image)
            weight_overlay = ndimage.morphology.distance_transform_edt(overlay_patch).astype(np.float64)
            max_value = np.partition(weight_overlay.flatten(), -2)[-2]
            weight_overlay /= max_value
            weight_overlay = np.clip(weight_overlay,0,1)
        else:
            weight_overlay = np.zeros_like(patch,dtype=np.float64)
        
        weight_no_overlay =np.ones_like(weight_overlay,dtype=np.float64)-weight_overlay
        #print(weight_overlay[0,:hide].astype(np.float16))
        # enhance_ratio = np.mean(image[np.where(overlay_patch==1)]) / np.mean(image[np.where(overlay_patch==0)])
        # print(enhance_ratio,np.mean(image[np.where(overlay_patch==1)]), np.mean(image[np.where(overlay_patch==0)]))
        # enhance_ratio = 1.001
        # weight_overlay *= enhance_ratio
        # weight_no_overlay[np.where(overlay_patch==1)] *= enhance_ratio
        #print(patch.shape,image.shape,weight_no_overlay.shape)
        #patch = image*weight_no_overlay
        patch = patch*weight_overlay + image*weight_no_overlay
        
    
        overlay_map[y0:y1,x0:x1] = 1
        #print(patch.dtype,output.dtype)
        output[y0:y1,x0:x1] = patch     # patch是float型而output是整型。请确保灰度级够大（不够大就乘个系数），否则这零点几的精度损失会带来明显的视觉差异（踩过坑）

    #     io.imsave('444_%d.tif'%i,output)
    io.imsave('changzi_stitch2.tif',output)

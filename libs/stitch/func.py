# coding: utf-8
from os import path

import numpy as np
from scipy import ndimage

from skimage import io




class StitchTool:
    def __init__(self,flat,bg) -> None:
        self.flat = flat
        self.bg = bg
    
    def set_flat(self,flat,bg):
        self.flat = flat
        self.bg = bg

    def correct(self,src,flat,bg,bias):

        print(bg.max(),bg.min())
        print(flat.max(),flat.min())
        flat = flat - bg + bias
        flat = flat.astype(np.float32)

        src = src.astype(np.float32)

        for i in range(src.shape[0]):
            raw = src[i,:,:]
            raw = 50*((raw-bg)/flat)
            src[i,:,:] = raw


        src = src-src.min()
        src = src.astype(np.uint16)
        return src
    
    def average_img(self,images,average_num):
        img_avg = np.zeros((images.shape[0]//average_num,images[1].images[2]),dtype=np.uint16)
        for i in range(images.shape[0]//average_num):
            img_avg[i] = np.mean(images[i*average_num:(i+1)*average_num],axis=0)
        return img_avg

    def stitch(self,img_path,save_path,grid_shape, average_num=1, bias=-1500, fill=True):
        
        [row_num,col_num] = grid_shape
        images = io.imread(img_path).astype(np.float32)


        images = self.correct(images, self.flat, self.bg, bias)


        if images.shape[0] != grid_shape[0]*grid_shape[1]*average_num:
            new_img = np.zeros((grid_shape[0]*grid_shape[1]*average_num, images.shape[1], images.shape[2]), dtype=np.uint16)
        if fill:
            print('图片缺帧，已在尾部补帧')
            new_img[:images.shape[0]] = images
        else:
            print('图片缺帧，已在头部补帧')
            new_img[-images.shape[0]:] = images
        images = new_img

        images = self.average_img(images,average_num)

        print(images.max()/65535.0) # TODO: 将放大系数30改为自适应系数，让灰度级别尽可能大点，但不要越界，即超过65535
        images = images/(images.max()/65535.0)*0.8
        images = images.astype(np.uint16)
        rows,cols = [],[]
        for row in range(row_num):
            rows += [row]*col_num
            if row%2==0:
                cols += list(range(col_num))
            else:
                cols += list(range(col_num-1,-1,-1))
    
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
            patch = patch*weight_overlay + image*weight_no_overlay
            
        
            overlay_map[y0:y1,x0:x1] = 1
            output[y0:y1,x0:x1] = patch     # patch是float型而output是整型。请确保灰度级够大（不够大就乘个系数），否则这零点几的精度损失会带来明显的视觉差异（踩过坑）
        print('>>>>  finish',save_path)
        io.imsave(save_path,output)
        return output

if __name__=='__main__':
    stitch('111_flat.tif',20,22)

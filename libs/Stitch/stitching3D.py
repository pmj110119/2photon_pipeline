import os
import numpy as np
from skimage import io
from stitching import config, main


def main_3D():
    config_list = [  # [mode, img_in_folder_path, img_out_path, avg_num, grid_shape[row, col]]
        ['mode1', r'E:\20210719宫颈样本3D\宫颈-150-PJ\CellVideo', r'E:\result.tif', 10, [12, 19]],
    ]

    for mode, img_in_folder_path, img_out_path, avg_num, grid_sh in config_list:
        row, col = grid_sh
        stitched_image_list = []

        cache_folder = 'temp'
        if not os.path.exists(os.path.join(img_in_folder_path, cache_folder)):
            os.mkdir(os.path.join(img_in_folder_path, cache_folder))
        img_depth = io.imread(os.path.join(img_in_folder_path, f'stitch-0\CellVideo\CellVideo 0.tif')).shape[0] // avg_num

        for i in range(img_depth):
            # Re-arrage images
            img2stitch_list = []
            for j in range(row * col):
                img_in_path = os.path.join(img_in_folder_path, f'stitch-{i}\CellVideo\CellVideo 0.tif')
                img2stitch_list.append(io.imread(img_in_path)[avg_num*img_depth:avg_num*img_depth+avg_num])
            io.imsave(np.array(os.path.join(img_in_folder_path, cache_folder, f'tmp-{i}.tif'), img2stitch_list))

            # 其实可以直接传数组，不需要来回读取，但是我懒得改了。下面是拼接的代码。
            img_in_path = os.path.join(img_in_folder_path, cache_folder, f'tmp-{i}.tif')
            # Update parameter
            img_avg_num = avg_num if avg_num else config['img_avg_num']
            grid_shape = grid_sh if grid_sh else config['grid_shape']

            overlap_row_avg1 = config['pre_overlap_row_1'][0:grid_shape[1]]
            overlap_row_avg2 = config['pre_overlap_row_2'][0:grid_shape[1]]
            pre_overlap_col = config['pre_overlap_col'][0:grid_shape[0]]

            # Stitch
            stitched_image = main(mode, img_in_path, img_out_path, smooth=True)
            stitched_image_list.append(stitched_image)

        # Save image
        io.imsave(img_out_path, np.array(stitched_image_list))


if __name__ == '__main__':
    main_3D()


[toc]

# Stitching Documentation

## File structure

```c++
-- stitching.py // 实现拼接功能的主要文件
-- configuration file.json // 拼接时的参数配置文件
-- update_parm.py // 更改系统时，重新标定数据的主要文件
```

## Requirements

```
numpy           1.19.3
opencv-python   4.4.0.46
scikit-image    0.16.2
scikit-learn    0.23.2
scipy           1.5.4
```

## How this algorithm works

### Stitch an TIF image

1. 由于有不同的S形拍图方式，所以利用row1_flip、row2_flip和col_flip将三个参数将所有拍摄方式转换成：左上=>右下S形拍摄（第一步向右）的方式。
2. 拼接时，首先一行一行拼成长条，随后把长条竖着拼接成为一张完整的大图

### Update parameters when system changes

## Usage

### Stitch an TIF image

按照如下步骤运行程序，对TIF图进行拼接。

1. 在stitching.py文件中修改`config_list`列表中数据
   1. `config_list`参数说明
      1. `config_list`中5个参数依次表示`mode`、`img_in_path`、`img_out_path`、`avg_num`和`grid_shape`
      2. `mode`参数详见Important Parameter Explanation
      3. `img_in_path`是输入图片的地址和文件名
      4. `img_out_path`是输出结果图片的地址和文件名
      5. `avg_num`是每多少张进行一次平均，如果该值为空，则填写`configuration file.json`中的默认值
      6. `grid_shape`是拼接的形状，如果该值为空，则填写`configuration file.json`中的默认值
   2. main函数参数说明
      1. `smooth`为缝合线模式
2. 运行stitching.py文件

### Update parameters when system changes

当成像系统发生改变后，需要对拼接参数重新标定。此时首先采集一张TIF拼接大图， 随后按照如下步骤，采用update_param.py文件进行参数标定。

1. 采集一张TIF拼接大图（建议$10 \times 10$以上）

2. 修改update_param.py文件中`img_in_path`为拼接大图路径

3. 运行update_param.py文件
   
   + 在console最后会输出类似如下数据：
   
   + ```
      --------- Summary ---------
      New pre_overlap_row_1 = [30.0, 36.0, 34.0, 36.0, 32.0, 33.0, 33.0, 41.0, 49.0]
      New pre_overlap_row_2 = [48.0, 33.0, 39.0, 41.0, 31.0, 34.0, 52.0, 48.0, 32.0]
      New pre_overlap_col   = [48.0, 45.0, 32.0, 40.0, 47.0, 50.0, 36.0, 40.0, 38.0]
      ```
      
   + 如果认为自动输出数据不准确，可以根据前面的输出手动计算，示例如下：
   
   + ```
      Row    1/  10 overlap = [31, 37, 38, 38, 34, 39, 35, 38, 86]
      Row    2/  10 overlap = [81, 44, 32, 77, 31, 32, 79, 38, 33]
      ...
      Column    1/  10 overlap = [30, 25, 29, 36, 65, 86, 31, 30, 34]
      Column    2/  10 overlap = [11, 31, 28, 44, 33, 40, 35, 23, 31]
      ...
      ```
   
4. 将新的`pre_overlap_row_1`、`pre_overlap_row_1`和`pre_overlap_col`填入`configuration file.json`中

## Important Parameter Explanation

+ stitching.py::mode
  + mode1：右下 => 左上，s形拍图，第一步往左移动
  + mode2：左下 => 右上，s形拍图，第一步往右移动
  + mode3：左上 => 右下，s形拍图，第一步往右移动
  + mode4：右上 => 左下，s形拍图，第一步往左移动

## Version & History

+ 20210116
  + 初版代码（陈晓程）
+ 20210527
  + 完成`updata_parm.py`，实现更换系统后的参数自动整定
  + 完善`stitching.py`，便于批量拼接
  + 编写`Readme.md`（此文档）
  + （孙羽轩）

---

### 光补偿

**light_compensation→method1:**

* bias1-3 ，enhance_per见注释
* 输入输出地址手动输入，输入为stack，输出为stack

### 参数更新

* 函数待整理，有需求联系我

### 注

* 拍图要从右下→左上拍摄，重叠量比较固定，没特殊需求不要采用其他拍摄方式。


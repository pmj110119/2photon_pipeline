### Update parameter

* overlap_limit：重叠量检测区间（0-512）
* img_avg_num：一个位置拍几张
* grid_shape：n*n的图

![image-20210524233008126](.assets/20210524235952.png)

1. False or True   #对折，一般不需要对折
2. fixedly or clockwise or anticlockwise  #一般为fixedly
3. S形拍图，（img_1f, img_2f）或者（img_2f, img_1f）
4. S形拍图，（img_2f, img_1f）或者（img_1f, img_2f）与3对应

![image-20210524233433400](.assets/20210524235956.png)

* 判断凸起的地方，应该是拐点，这里需要改进。。。（有这样的凸起为正常）

![image-20210524234249349](.assets/20210524235959.png)

![image-20210524234329633](.assets/20210525000003.png)

* 单调的可能是空白或者图像质量太差也可能是异常，就上面旋转啥的错了。

![image-20210524234447310](.assets/20210525000007.png)



* 所以需要拍大图，对所有奇数行偶数行计算所得重叠量进行异常值剔除（可能是空白导致错也可能是拐点判断错误）